import os
import blifparser.blifparser as bparser
from pyanvil import world
import numpy as np
import sys
import random

import itertools
import copy

from queue import PriorityQueue

import matplotlib.pyplot as plt

air = 0
block = 1
dust = 2
torch = 3
lever = 4
light = 5
rblock = 6

world_size = 25

block_mapping = {
    air: ('minecraft:air', {}),
    block: ('minecraft:iron_block', {}),
    rblock: ('minecraft:gold_block', {}),
    dust: ('minecraft:redstone_wire', {}),
    torch: ('minecraft:redstone_torch', {}),
    lever: ('minecraft:lever', {}),
    light: ('minecraft:redstone_lamp', {})
}

wool_blocks = [
    'minecraft:white_wool',
    'minecraft:orange_wool',
    'minecraft:magenta_wool',
    'minecraft:cyan_wool'
]

def iter_3d(x, y, z):
    return itertools.product(range(x), range(y), range(z))

class FU(object):

    def __init__(self, inputs, outputs, layout):
        self.layout = layout
        self.inputs = inputs
        self.outputs = outputs

    def get_dims(self):
        return np.array((len(self.layout[0][0]), len(self.layout), len(self.layout[0])))

    def place_FU(self, position, mc_world):

        dims = list(self.get_dims())
        for (x, y, z) in iter_3d(dims[0], dims[1], dims[2]):
            pos = (position[0] + x, position[1] + y, position[2] + z)
            element = self.layout[y][z][x]
            if isinstance(element, tuple):
                block_id = block_mapping[element[0]][0]
                tags = element[1]
            else:
                block_id = block_mapping[element][0]
                tags = block_mapping[element][1]
            mc_world.get_block(pos).set_state(
                world.BlockState(block_id, tags))

class FUInst(object):

    def __init__(self, FU, position, IOmapping, movable=False, name=""):
        self.FU = FU
        self.position = position
        self.IOmapping = IOmapping
        self.movable = movable
        self.name = name

class TopLevel(object):

    def __init__(self, world_size):
        self.world_size = world_size
        self.grid = [
            [
                [
                    air for _ in range(world_size[0])
                ]
                for _ in range(world_size[2])
            ] for _ in range(world_size[1])
        ]

        self.route_grid = [
            [
                [
                    0 for _ in range(world_size[0])
                ]
                for _ in range(world_size[2])
            ] for _ in range(world_size[1])
        ]

        self.wire_mapping = {}
        self.wire_idx = 2

        self.routing_start = {}
        self.FU_list = []

    def set_grid(self, pos, val):
        self.grid[pos[1]][pos[2]][pos[0]] = val

    def get_grid(self, pos):
        return self.grid[pos[1]][pos[2]][pos[0]]

    def set_routing(self, pos, val):
        self.route_grid[pos[1]][pos[2]][pos[0]] = val

    def get_routing(self, pos):
        # if grid is None:
        #     grid = self.route_grid
        return self.route_grid[pos[1]][pos[2]][pos[0]]

    def get_routing_below(self, pos, grid=None):
        if grid is None:
            grid = self.route_grid
        return grid[pos[1]][pos[2]][pos[0]]

    def push_wire_mapping(self, wire_name):
        self.wire_mapping[wire_name] = self.wire_idx
        self.wire_idx += 1

    def register_FU(self, inst):
        print(f"Inserting FU with {inst.movable} {inst.name}")
        self.FU_list.append(inst)

    def get_bounding_box(self):
        bot_left = [self.world_size[0], self.world_size[1], self.world_size[2]]
        top_right = [0, 0, 0]
        for (x, y, z) in iter_3d(self.world_size[0], self.world_size[1], self.world_size[2]):
            if(self.get_grid((x, y, z)) != air):
                if(x < bot_left[0]):
                    bot_left[0] = x
                if(x > top_right[0]):
                    top_right[0] = x
                if(y < bot_left[1]):
                    bot_left[1] = y
                if(y > top_right[1]):
                    top_right[1] = y
                if(z < bot_left[2]):
                    bot_left[2] = z
                if(z > top_right[2]):
                    top_right[2] = z
        return (bot_left, top_right)

    def calc_energy(self):
        metric = 'volume'

        if metric == 'LJ':
            sigma = 50
            epsilon = 2
            s = 0
            for i in range(len(self.FU_list)):
                for j in range(i):
                    disp = list(np.array(self.FU_list[0].position) - np.array(self.FU_list[1].position))
                    dist2 = sum(x * x for x in disp)
                    if(dist2 == 0):
                        continue
                    r_inv = (sigma / dist2) ** 3
                    s += 4 * epsilon * r_inv * (r_inv - 1)
            return s
        elif metric == 'volume':
            bot_left, top_right = self.get_bounding_box()
            print(f"Bouding volume: {bot_left}, {top_right}")
            return (top_right[0]-bot_left[0]) * (top_right[1]-bot_left[1]) * (top_right[2]-bot_left[2])

        else:
            print("Invaid metric")
            sys.exit(1)

    # Places a FU on both the world grid as well as the routing grid
    def place_FU_grid(self, fu_inst):

        fu = fu_inst.FU
        position = fu_inst.position
        inout_mapping = fu_inst.IOmapping

        dims = list(fu.get_dims())
        # grid_dims = (len(grid[0][0]), len(grid[0]), len(grid)) #TODO Should check out of bounds
        for (x, y, z) in iter_3d(dims[0], dims[1], dims[2]):
            pos = (position[0] + x, position[1] + y, position[2] + z)
            self.set_grid(pos, fu.layout[y][z][x])
            if fu.layout[y][z][x] != air:
                # self.route_grid[pos[1]][pos[2]][pos[0]] = -1
                self.set_routing(pos, -1)

        '''
        for port, offset in fu.outputs.items():
            if port in inout_mapping:
                wire_name = inout_mapping[port]
                pos = (position[0] + offset[0], position[1] + offset[1], position[2] + offset[2])
                print(f"Wire {str(wire_name)} at {str(pos)} needs input")
        '''
        # Setup input/output ports on the grid
        for (port, offset) in itertools.chain(fu.inputs.items(), fu.outputs.items()):
            if port not in inout_mapping:
                print(f"Port {str(port)} not in inout mapping, skipping")
                continue
            wire_name = inout_mapping[port]
            if wire_name not in self.wire_mapping:
                print(f"Wire {str(wire_name)} not in mapping, creating new idx {str(self.wire_idx)}")
                self.push_wire_mapping(wire_name)

            wire_num = self.wire_mapping[wire_name]
            # print(f"Mapping wire {str(wire_name)} to idx {str(wire_num)}")
            pos = (position[0] + offset[0], position[1] + offset[1], position[2] + offset[2])
            # self.route_grid[pos[1]][pos[2]][pos[0]] = wire_num
            self.set_routing(pos, wire_num)
            if(self.grid[pos[1]-1][pos[2]][pos[0]] != rblock):
                print(f"Port {str(port)} must end on routing block")
                sys.exit(1)
            else:
                self.route_grid[pos[1]-1][pos[2]][pos[0]] = -wire_num

        for port, offset in fu.inputs.items():
            if port in inout_mapping:
                wire_name = inout_mapping[port]
                pos = (position[0] + offset[0], position[1] + offset[1], position[2] + offset[2])
                self.routing_start.update({wire_name: pos})

    def erase_FU(self, fu_inst):
        dims = list(fu_inst.FU.get_dims())
        position = fu_inst.position
        for (x, y, z) in iter_3d(dims[0], dims[1], dims[2]):
            pos = (position[0] + x, position[1] + y, position[2] + z)
            self.set_routing(pos, 0)
            self.set_grid(pos, 0)

    def is_spot_free(self, fu, position):
        dims = list(circuit.get_dims())
        padding = 4
        upper_dims = (self.world_size[0] - 2*padding, self.world_size[1] - 2*padding, self.world_size[2] - 2*padding)
        origin = np.random.rand(3) * (np.array(upper_dims) - dims + np.array([padding, padding, padding]))
        origin = origin.astype(int)
        is_clear = True
        for (x, y, z) in iter_3d(dims[0], dims[1], dims[2]):
            if self.grid[origin[1]+y][origin[0]+x][origin[2]+z] != 0:
                is_clear = False
                break
        return is_clear


    def find_free_spot(self, fu):
        found = False
        attempts = 0
        max_attempts = 100
        while not found and attempts < max_attempts:
            # Find random spot where it will fit
            dims = list(circuit.get_dims())
            padding = 4
            upper_dims = (self.world_size[0] - 2*padding, self.world_size[1] - 2*padding, self.world_size[2] - 2*padding)
            origin = np.random.rand(3) * (np.array(upper_dims) - dims + np.array([padding, padding, padding]))
            origin = origin.astype(int)
            is_clear = True
            for (x, y, z) in iter_3d(dims[0], dims[1], dims[2]):
                if self.grid[origin[1]+y][origin[0]+x][origin[2]+z] != 0:
                    is_clear = False

            if is_clear:
                found = True
            attempts += 1
        if found:
            return origin
        else:
            print("Unable to find free random spot")
            return None

    def do_place_route(self):
        # Make sure initial config works, getting initial energy
        r = self.do_routing()
        energys = []
        accept_ratio = []
        swap_ratio = []
        move_ratio = []
        accepted = 0
        routing_energy = []
        volume_energy = []
        swaps_accepted = 0
        moves_accepted = 0

        if not r[0]:
            print("Can not route initial placement, retry")
        else:
            # Calculate routing pressure
            route_length = 0
            for wire_name, new_path in r[1]:
                for pathway in new_path:
                    route_length += 1
            curr_energy = self.calc_energy()

            print(f"Initial energy {curr_energy} routing {route_length}")
            curr_energy += route_length ** 2

            for iteration in range(100):
                print(f"Iteration {iteration}")
                energys.append(curr_energy)
                volume_energy.append(curr_energy - (route_length**2))
                routing_energy.append(route_length**2)

                # Pick a random circuit and propose a move
                FU_inst = self.FU_list[int(np.random.rand(1) * len(self.FU_list))]
                while not FU_inst.movable:
                    FU_inst = self.FU_list[int(np.random.rand(1) * len(self.FU_list))]
                print(f"Moving {FU_inst.name}")

                if(random.random() < 0.25):
                    # Swapping algo
                    FU_inst_2 = self.FU_list[int(np.random.rand(1) * len(self.FU_list))]
                    while not FU_inst_2.movable:
                        FU_inst_2 = self.FU_list[int(np.random.rand(1) * len(self.FU_list))]
                    print(f"Swapping with {FU_inst_2.name}")

                    old_location = FU_inst.position
                    old_location_2 = FU_inst_2.position

                    self.erase_FU(FU_inst)
                    self.erase_FU(FU_inst_2)
                    should_revert = False
                    if(self.is_spot_free(FU_inst_2, old_location) and
                            self.is_spot_free(FU_inst, old_location_2)):

                        try:
                            FU_inst.position = old_location_2
                            self.place_FU_grid(FU_inst)

                            FU_inst_2.position = old_location
                            self.place_FU_grid(FU_inst)

                            new_energy = self.calc_energy()
                            r = self.do_routing(curr_energy - new_energy)
                            if not r[0]:
                                print("Could not swap, reverting")
                                should_revert = True
                            else:
                                route_length = 0
                                for wire_name, new_path in r[1]:
                                    for pathway in new_path:
                                        route_length += 1
                                new_energy += route_length ** 2

                                if new_energy < curr_energy:
                                    print("Accepting swap")
                                    accepted += 1
                                    swaps_accepted += 1
                                else:
                                    should_revert = True
                        except IndexError:
                            should_revert = True

                    else:
                        should_revert = True
                        print("Swap location not free, reverting")
                    if should_revert:
                        self.erase_FU(FU_inst)
                        self.erase_FU(FU_inst_2)

                        FU_inst.position = old_location
                        FU_inst_2.position = old_location_2

                        self.place_FU_grid(FU_inst)
                        self.place_FU_grid(FU_inst_2)
                else:
                    fu = FU_inst.FU
                    old_location = FU_inst.position

                    # Try to replace the FU
                    self.erase_FU(FU_inst)
                    # new_spot = self.find_free_spot(fu)
                    # if new_spot is None:
                    #     continue
                    bounding_box = self.get_bounding_box()
                    side_lens = list(np.array(bounding_box[0]) - np.array(bounding_box[1]))
                    offset = [random.gauss(0, side_lens[d]) for d in range(3)]

                    new_spot = list((np.array(old_location) + np.array(offset)).astype(int))

                    dims = list(fu.get_dims())
                    FU_inst.position = new_spot

                    while any(new_spot[d] < 0 or new_spot[d] >= self.world_size[d]-dims[d] for d in range(3)) or not self.is_spot_free(fu, new_spot):
                        offset = [random.gauss(0, side_lens[d]) for d in range(3)]
                        new_spot = list((np.array(old_location) + np.array(offset)).astype(int))
                        FU_inst.position = new_spot

                    FU_inst.position = new_spot
                    self.place_FU_grid(FU_inst)
                    # See if the move would reduce energy
                    new_energy = self.calc_energy()

                    should_revert = False

                    prob_accept = False
                    if(new_energy > curr_energy):
                        prob = (np.e ** (curr_energy - new_energy))
                        print(prob)
                        if(random.random() < prob):
                            prob_accept = True

                    if((new_energy < curr_energy) or prob_accept):
                        if prob_accept:
                            print(f"Move from {old_location} to {new_spot} increased energy from {curr_energy} to {new_energy}, but accepting anyways")

                        else:
                            print(f"Move from {old_location} to {new_spot} reduced energy from {curr_energy} to {new_energy}, half accepting")

                        # Check if we can route
                        r = self.do_routing(curr_energy - new_energy)
                        if not r[0]:
                            print("Could not route, reverting")
                            should_revert = True
                        else:
                            # Calculate routing pressure
                            route_length = 0
                            for wire_name, new_path in r[1]:
                                for pathway in new_path:
                                    route_length += 1
                            new_energy += route_length ** 2
                            if(new_energy > curr_energy):
                                prob = (np.e ** ((curr_energy - new_energy)/(curr_energy/9)))
                                print(prob)
                                if(random.random() < prob):
                                    prob_accept = True
                            if((new_energy < curr_energy) or prob_accept):
                                print(f"Move with routing reduced energy from {curr_energy} to {new_energy}, accepting")
                                curr_energy = new_energy
                            else:
                                print(f"Routing increased energy from {curr_energy} to {new_energy}, rejecting")
                                should_revert = True
                    else:
                        print(f"Move from {old_location} to {new_spot} increased energy from {curr_energy} to {new_energy}, rejecting")
                        should_revert = True

                    if should_revert:
                        self.erase_FU(FU_inst)
                        FU_inst.position = old_location
                        self.place_FU_grid(FU_inst)
                    else:
                        accepted += 1
                        moves_accepted += 1

                if iteration == 50:
                    old_grid = copy.copy(self.grid)
                    self.do_routing(write_routing=True)
                    top_lvl.write_to_world((0, 50, 0), "../../../../.minecraft/saves/test_gen2")
                    self.grid = old_grid

                if iteration == 0:
                    old_grid = copy.copy(self.grid)
                    self.do_routing(write_routing=True)
                    top_lvl.write_to_world((0, 50, 100), "../../../../.minecraft/saves/test_gen2")
                    self.grid = old_grid

                accept_ratio.append(accepted / (iteration + 1))
                swap_ratio.append(swaps_accepted / (iteration + 1))
                move_ratio.append(moves_accepted / (iteration + 1))

            plt.scatter(s=20, x=list(range(len(energys))), y=energys, label="Total")
            plt.scatter(s=5, x=list(range(len(energys))), y=routing_energy, label="Routing")
            plt.scatter(s=5, x=list(range(len(energys))), y=volume_energy, label="Volume")
            plt.xlabel('Iteration')
            plt.ylabel('Energy')
            plt.legend()
            plt.show()
            plt.scatter(x=list(range(len(accept_ratio))), y=accept_ratio)
            plt.scatter(x=list(range(len(accept_ratio))), y=swap_ratio)
            plt.scatter(x=list(range(len(accept_ratio))), y=move_ratio)
            plt.xlabel('Iteration')
            plt.ylabel('Acceptance Ratio')
            plt.show()
            print("Done optimizing, final routing")
            self.do_routing(write_routing=True)

    def write_to_world(self, origin, path):
        clear_size = max(self.world_size) * 2
        air_layout = [[[air] * clear_size] * clear_size] * clear_size
        clear_FU = FU(None, None, air_layout)
        total_FU = FU(None, None, self.grid)

        abs_path = os.path.abspath(path)
        print(f"Outputing world to {str(abs_path)}")

        # print(self.route_grid)

        with world.World(abs_path) as mc_world:
            clear_FU.place_FU(origin, mc_world)
            total_FU.place_FU(origin, mc_world)

    def do_routing(self, write_routing=False, verbose=False, max_energy=0):

        routing_items = list(self.routing_start.items())
        attempts = 0
        max_attempts = len(routing_items)

        test_grid = copy.deepcopy(self.route_grid)
        paths = []

        while attempts < max_attempts:

            route_length = 0
            could_route = True
            paths = []
            test_grid = copy.deepcopy(self.route_grid)
            random.shuffle(routing_items)

            if verbose:
                print(f"Starting routing at attempt {str(attempts)}/{str(max_attempts)}")
                print(f"Routing order: {str([str(x[0]) for x in routing_items])}")
            for (wire, start) in routing_items:
                could_route = self.route_wire(wire, start, test_grid, max_energy=max_energy)
                if not could_route[0]:
                    print(f"Routing failed on {str(wire)}")
                    could_route = False
                    break
                else:
                    paths.append((wire, could_route[1]))
                    route_length += len(could_route[1])
                    if(max_energy != 0 and route_length ** 2 > max_energy):
                        print("Stopping routing due to exceeding energy")
                        could_route = False
                        break
            if could_route:
                print("Done routing")
                break

            attempts += 1

        if write_routing:
            for wire_name, new_path in paths:
                for pathway in new_path:
                    self.grid[pathway[1]][pathway[2]][pathway[0]] = dust
                    self.grid[pathway[1]-1][pathway[2]][pathway[0]] = rblock

        if(attempts == max_attempts):
            print("Unable to route")
            return (False, None)
        else:
            return (True, paths)

    def setup_routing(self):
        flat_offsets = [
            (+1, +0, +0),
            (-1, +0, +0),
            (+0, +0, +1),
            (+0, +0, -1),

            (+1, +1, +0),
            (+1, -1, +0),

            (-1, +1, +0),
            (-1, -1, +0),

            (+0, +1, +1),
            (+0, -1, +1),

            (+0, +1, -1),
            (+0, -1, -1)
        ]

        to_check = set()
        intersect = [[]]
        all_checks = []

        self.check_dict = {}

        for n in flat_offsets:
            self.check_dict[n] = []
            for dx, dy, dz in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
                pos = (n[0] + dx, n[1] + dy, n[2] + dz)
                to_check.add(pos)
                intersect[-1].append(pos)
                all_checks.append(pos)
                self.check_dict[n].append(pos)
            intersect.append([])

        intersect.pop()

        # print(len(all_checks))
        # inter = set([x for x in all_checks if all(x in sub for sub in intersect)])
        # print(inter)
        # print(len(inter))

        # print(self.check_dict)

    def in_world(self, pos):
        return all(pos[d] >= 0 and pos[d] < self.world_size[d] for d in range(3))

    def is_postive(self, pos):
        return all(d >= 0 for d in pos)

    def get_neighbours(self, pos, wire_idx, routing_grid):

        outputs = []
        for offset, to_check in self.check_dict.items():
            pos_offset = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
            try:
                if(any(d < 0 for d in pos_offset)):
                    continue
                if(self.get_routing(pos_offset) in [wire_idx]):
                    outputs.append(pos_offset)
                else:
                    can_add = True
                    for delta in to_check:
                        check_delta = (pos_offset[0] + delta[0], pos_offset[1] + delta[1], pos_offset[2] + delta[2])
                        if self.get_routing(check_delta) not in [0, wire_idx, -wire_idx]:
                            can_add = False
                            break
                    if can_add:
                        outputs.append(pos_offset)
            except IndexError:
                continue

        # print(outputs)
        return outputs
        # neighbours = [
            # (pos[0]+1, pos[1]+0, pos[2]+0),
            # (pos[0]-1, pos[1]+0, pos[2]+0),
            # (pos[0]+0, pos[1]+0, pos[2]+1),
            # (pos[0]+0, pos[1]+0, pos[2]-1),

            # (pos[0]+1, pos[1]+1, pos[2]+0),
            # (pos[0]+1, pos[1]-1, pos[2]+0),

            # (pos[0]-1, pos[1]+1, pos[2]+0),
            # (pos[0]-1, pos[1]-1, pos[2]+0),

            # (pos[0]+0, pos[1]+1, pos[2]+1),
            # (pos[0]+0, pos[1]-1, pos[2]+1),

            # (pos[0]+0, pos[1]+1, pos[2]-1),
            # (pos[0]+0, pos[1]-1, pos[2]-1)
        # ]

        # # Filter in world bounds
        # valid_neighbours = filter(
            # lambda n: all(n[d] >= 0 and n[d]+1 < self.world_size[d] for d in range(3)),
            # neighbours
        # )

        '''
        clean_neighbours = filter(
            lambda n: all(
                all(n[d] + offset[d] >= 0 and n[d] + offset[d] < self.world_size[d] for d in range(3))
                and self.route_grid[n[1] + offset[1]][n[2] + offset[2]][n[0] + offset[0]] in [0, wire_idx, -wire_idx] for
                offset in itertools.product([0], [-1, 0, 1], [0])),
            valid_neighbours
        )
        return clean_neighbours
        '''
        # offsets = [(-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)]
        # clean_neighbours = []
        # for n in valid_neighbours:
            # if routing_grid[n[1]][n[2]][n[0]] in [wire_idx, -wire_idx]:
                # clean_neighbours.append(n)
                # continue
            # valid = True
            # for dx, dy, dz in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
            # # for dx, dy, dz in offsets:
                # pos = (n[0] + dx, n[1] + dy, n[2] + dz)
                # if routing_grid[pos[1]][pos[2]][pos[0]] not in [0, wire_idx, -wire_idx]:
                    # valid = False
                    # # print(f"Skipping {str(n)} offset {str((dx, dy, dz))} due to {str(self.route_grid[pos[1]][pos[2]][pos[0]] )}")
            # if valid:
                # clean_neighbours.append(n)
        # # print(clean_neighbours)
        # return clean_neighbours
        '''
        clean_neighbours = []
        for n in valid_neighbours:
            pos = (n[0], n[1], n[2])
            if self.route_grid[pos[1]][pos[2]][pos[0]] in [0, wire_idx, -wire_idx]:
                clean_neighbours.append(n)

        return clean_neighbours
        '''

    def route_wire(self, wire_name, start, routing_grid, max_energy=0):
        if wire_name not in self.wire_mapping:
            print(f"Wire {str(wire_name)} not in mapping, skipping")
        wire_num = self.wire_mapping[wire_name]
        print(f"Routing wire {str(wire_name)} with idx {str(wire_num)} from {str(start)}")

        # BFS shortest path from
        # https://www.geeksforgeeks.org/building-an-undirected-graph-and-finding-shortest-path-using-dictionaries-in-python/
        queue = [[start]]
        explored = set()

        while queue:
            path = queue.pop(0)
            node = path[-1]

            if node not in explored:
                neighbours = self.get_neighbours(node, wire_num, routing_grid)
                # print(f"N list: {str(list(neighbours))}")
                for neighbour in list(neighbours):
                    if(start == neighbour or neighbour == (start[0], start[1]-1, start[2])):
                        continue

                    if (neighbour[0], neighbour[1]+1, neighbour[2]) in path:
                        continue

                    if (neighbour[0], neighbour[1]-1, neighbour[2]) in path:
                        continue

                    new_path = list(path)
                    new_path.append(neighbour)
                    # print(f"Test path: {str(new_path)}")
                    if(max_energy == 0 or len(new_path) ** 2 < max_energy):
                        queue.append(new_path)

                    if self.get_routing(neighbour) in [wire_num, -wire_num]:
                        print(f"Found path for {str(wire_name)}")
                        print(new_path)
                        for pathway in new_path:
                            # self.grid[pathway[1]][pathway[2]][pathway[0]] = dust
                            # self.grid[pathway[1]-1][pathway[2]][pathway[0]] = rblock

                            routing_grid[pathway[1]][pathway[2]][pathway[0]] = wire_num
                            routing_grid[pathway[1]-1][pathway[2]][pathway[0]] = -wire_num
                        return (True, new_path)
                explored.add(node)
        print(f"Couldnt route {wire_name}")
        return (False, None)


# Cell library layouts

west_east = {'west': 'side', 'east': 'side'}
north_south = {'north': 'side', 'south': 'side'}

inverter_layout = [
    [[rblock, block, block, block]],
    [[(dust, west_east), (dust, west_east), block, rblock]],
    [[air, air, torch, (dust, west_east)]]
]

nand_layout = [
    [[rblock, block, rblock],
     [block, block, block],
     [block, rblock, block]],
    [[(dust, north_south), air, (dust, north_south)],
     [block, block, block],
     [air, (dust, {'north': 'up'}), air]],
    [[air, air, air],
     [torch, (dust, {'west': 'side', 'east': 'side', 'south': 'side'}), torch],
     [air, air, air]],
]

input_layout = [
    [[block, block, rblock]],
    [[(lever, {'facing': 'west'}), block, dust]],
    [[air, light, air]]
]

output_layout = [
    [[rblock, block]],
    [[(dust, {'west': 'side'}), light]],
    [[air, air, (dust, west_east)]]
]

inverter = FU({'A': (0, 1, 0)}, {'Y': (3, 2, 0)}, inverter_layout)
nand = FU({'A': (0, 1, 0), 'B': (2, 1, 0)}, {'Y': (1, 1, 2)}, nand_layout)

inpt = FU({}, {'Y': (2, 1, 0)}, input_layout)
outpt = FU({'A': (0, 1, 0)}, {}, output_layout)

FU_map = {
    'NOT': inverter,
    'NAND': nand}

# Parse blif
path = os.path.abspath("./synth.blif")
print(f"Reading blif at {str(path)}")
parser = bparser.BlifParser(path)
blif = parser.blif

print(f"Top level module {blif.model.name}")
print(f"Top level inputs: {str(blif.inputs.inputs)}")
print(f"Top level outputs: {str(blif.outputs.outputs)}")

top_lvl = TopLevel((world_size, world_size, world_size))

print("\nPlacing sub circuits:")
for subcircuit in blif.subcircuits:
    # print(subcircuit.__dict__)
    params = {param.split('=')[0]: param.split('=')[1] for param in subcircuit.params}
    circuit = FU_map[subcircuit.modelname]
    origin = top_lvl.find_free_spot(circuit)
    print(f"Placing {subcircuit.modelname} at {str(origin)}")

    inst = FUInst(circuit, origin, params, movable=True, name=subcircuit.modelname)
    top_lvl.place_FU_grid(inst)
    top_lvl.register_FU(inst)

print("\nPlacing Inputs and Outputs:")
for top_input in blif.inputs.inputs:
    circuit = inpt
    origin = top_lvl.find_free_spot(circuit)
    print(f"Placing input at {str(origin)}")
    params = {'Y': top_input}
    inst = FUInst(circuit, origin, params, movable=True, name="Input")
    top_lvl.place_FU_grid(inst)
    top_lvl.register_FU(inst)

for top_output in blif.outputs.outputs:
    circuit = outpt
    origin = top_lvl.find_free_spot(circuit)
    print(f"Placing output at {str(origin)}")
    params = {'A': top_output}
    inst = FUInst(circuit, origin, params, movable=True, name="Output")
    top_lvl.place_FU_grid(inst)
    top_lvl.register_FU(inst)

print("\nRouting:\n")
print(f"\nInitial Energy: {str(top_lvl.calc_energy())}:\n")

top_lvl.setup_routing()
# sys.exit(1)

top_lvl.do_place_route()

top_lvl.write_to_world((200, 10, 0), "../../../../.minecraft/saves/test_gen2")
