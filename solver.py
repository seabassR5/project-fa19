import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from utils import *

from student_utils import *
def soda_drop_off(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    car_loc = list_of_locations.index(starting_car_location)
    car_stayed_parked = [car_loc for _ in range (2)]
    homes = [i for i in range(len(list_of_locations)) if list_of_locations[i] in list_of_homes]
    rao_didnt_drive_anyone_home = { car_loc : homes }
    return car_stayed_parked, rao_didnt_drive_anyone_home

    pass

# def pruned_cluster(key, val, list_of_homes_by_ndx):

def prune_all_non_homes(clusters, list_of_homes_by_ndx):
    """
    clusters: dict with key val as indecies
    list_of_homes: list of the indexes of homes

    return val: only clusters that contain homes, non home locations are removed from list, clusters with empty are removed
    """
    new_clusters 
    homes = set(list_of_homes_by_ndx)

    for bus_stop, locations in clusters:
        locations.append(bus_stop) #not sure if cluster includes the bus_stop
        drop_offs = homes.intersection(set(locations)) 
        ta_cluster = list(drop_offs)
        if len(new_cluster) > 0:
            new_clusters[bus_stop] = ta_cluster
    return new_clusters


def medoids_solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    GERERATING CLUSTERS/CLUSTERINGS:
        # want to make multiple maps of {bus_stop: [list of Locations (cluster, mix of TA/nonTA)], ...}
           each with varying amount of K clusters, say from 1 to len(locatoins)
           WE NEED to prune clusters and remove any non TA locations, IF the list becomes empty remove it
           pruned_cluster = { key, [loc for loc in val if list_of_locations[loc] in list_of_homes] for (key, val) in clusters}
        
        
        # generate the Sortest paths graph for all pairs from adjacency graph then pass to nx.clustering/medoids
        # the output of this function wil be REALLY CLOSE to 
            the drop-off locatons dict we want to return, 
            just remove any non-TA locations
        
         

        # maybe we should somhow only consider clusters of TA home locations rather than passing the whole graph in
     
    SCOPE TSP: lookup mlrose library? 
        https://towardsdatascience.com/solving-travelling-salesperson-problems-with-python-5de7e883d847
    TSP ON CLUSTERS: we will run the following on all the clusterings
        run some tsp alg on the shortest path distance matrix considering only the bus_stops/drop_off_loc
        then we will get some ordering of the bus_stops -> bus_stop_list = [3, 5, 2, 8, 3], that corespond to location
        
        HOWEVER:
            we need the locations in-between -> [3, ???, 5, ???, ???, 2, ???, 8, 3]
        SO:
            we get the shortest path list between each bus_stop i and i+1 with:
            nx.single_source_dijkstra(G, source, target=None, cutoff=None, weight='weight')
            https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_dijkstra.html#networkx.algorithms.shortest_paths.weighted.single_source_dijkstra 
            which returns tuple: distance, path
            AND insert the path in the bus_stop_list
            full_path = []
            for ndx in range(len(bus_stop_list) - 1):
                bus_stop = bus_stop_list[ndx]
                len, path = single_source_dijkstra(G, stop, bus_stop_list[ndx+1])
                full_path.append(bus_stop)
                full_path.append(path)
        FINALLY:
            after insterting all the locations inbetween the bus stops,
            this will be the car route
    triky bit:
        we will have the cluster locations, and shortest path distances between them,
        we want to return the tsp path but we only have bus_stops and not the locations inbetween
    
    """

    pass

"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    return soda_drop_off(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params)

    pass

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
