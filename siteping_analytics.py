from functools import total_ordering

import networkx
import numpy
import pandas as pd
import numpy as np
from scipy.stats import kruskal, stats
from sklearn.cluster import AgglomerativeClustering
import argparse
import os
import us
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", default=2, type=int, help="Maximum z score for a pair of servers. Default is 2")
    parser.add_argument("-v", default=0.1, type=float, help="Maximum coefficient of variation for a point of measurement between a pair of servers. Default is 0.1")
    parser.add_argument("-c", default="rtt", type=str, help="rtt or rtt_normalized. Default is rtt")
    parser.add_argument("-w", default=None, type=str, help="Path to a weights file. Default is  None")
    args = parser.parse_args()

    column = args.c

    print("Generating results...")

    rawDf = pd.read_json('siteping_cleaned3_pretty.json')
    df = rawDf.loc[rawDf['isMobile'] == False]
    #df = pd.read_pickle(getDataFrame())

    if args.w is not None:
        if os.path.isfile(args.w):
            weights = pd.read_csv(args.w)
            weights['authoritative_state'] = weights['state'].map(us.states.mapping('name', 'abbr'))
            weights = weights[['authoritative_state', 'value']]
            weights.value = weights.value.astype(int)

            df = df.reset_index().merge(weights, how='inner')
            df['median'] = df['median'] * df['value']

            del df['value']

            df = df.set_index(['recursive_state', 'authoritative_state', 'recursive_ip', 'authoritative_ip'])
        else:
            args.w = None

    # Create main results directory if it doesn't exist
    if not os.path.isdir('./results'):
        os.mkdir('./results')

    # Create results directory if it doesn't exist
    if args.w is not None:
        results_dir = './results/results_{}_{}_{}_{}/'.format(args.c, args.z, args.v,  os.path.basename(args.w))
    else:
        results_dir = './results/results_{}_{}_{}_{}/'.format(args.c, args.z, args.v,  "None")
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Generating rankings, along with the count for each state

    state_rankings = df.groupby(['state'])[['rtt']].agg([np.median, 'count'])
    state_rankings.columns = state_rankings.columns.droplevel()
    state_rankings = state_rankings.sort_values('median')

    state_rankings['rank'] = state_rankings['median'].rank(method='max')

    state_to_state_p_values = pd.DataFrame()
    state_to_state_p_values_i = pd.DataFrame()
    state_to_state_h_values = pd.DataFrame()

    # Generate kruskals values for each state pair
    for i in range(0, state_rankings.shape[0]):
        s1 = list(pd.DataFrame(state_rankings.iloc[[i]]).index)[0]

        for j in range(0, state_rankings.shape[0]):
            s2 = list(pd.DataFrame(state_rankings.iloc[[j]]).index)[0]
            h, p = kruskal(df.loc[df['state'] == s1]['rtt'], df.loc[df['state'] == s2]['rtt'])

            state_to_state_p_values.at[s1, s2] = p
            state_to_state_p_values_i.at[s1, s2] = 1 - p
            state_to_state_h_values.at[s1, s2] = h

    # Generate clusters based on inverted kruskal's values
    ac = AgglomerativeClustering(distance_threshold=.5, n_clusters=None, compute_full_tree=True).fit(state_to_state_p_values_i)
    state_rankings['cluster'] = ac.labels_

    # Save output
    state_rankings.to_csv('{}/state_rankings.csv'.format(results_dir))
    state_to_state_p_values.to_csv('{}/state_to_state_p_values.csv'.format(results_dir))
    state_to_state_h_values.to_csv('{}/state_to_state_h_values.csv'.format(results_dir))

    # Graph analysis
    pValueDf = pd.read_csv('{}/state_to_state_p_values.csv'.format(results_dir))
    pValueDf = pValueDf.rename(columns={'Unnamed: 0': 'states'})
    pValueDf = pValueDf.set_index(['states'])
    pValueDf[pValueDf > 0.05] = 1
    pValueDf[pValueDf <= 0.05] = 0

    # Create a network from the data
    net = networkx.from_pandas_adjacency(df)

    # Find cliques and convert to objects
    raw_cliques = list(networkx.algorithms.community.greedy_modularity_communities(net))
    cliques = [CliqueOfStates(i, l, [state_rankings.loc[state]['rank'] for state in l],
                              [state_rankings.loc[state]['median'] for state in l])
               for i, l in enumerate(raw_cliques)]
    cliques = sorted(cliques)

    i = 0
    colors = list(mcolors.TABLEAU_COLORS)
    # Print cliques and generate intra-clique CDFs
    for clique in cliques:
        clique.set_id(i)

        j = 0
        for state in clique.states:
            x = df.loc[pd.IndexSlice[state], :]['rtt']
            color = colors[j % len(colors)]
            plt.hist(x, bins=400, density=True, cumulative=True, label="{} ({:n})".format(state, clique.ranks[j]),
                     histtype='step', alpha=0.8, color=color, range=(0, 100))
            j += 1

        plt.legend(loc='upper left')
        plt.savefig('{}/cluster_{}_cdf.png'.format(results_dir, i))
        plt.show()

        i += 1
        print(clique)
        print()

    # Generate a CDF for each clique (all cliques)
    for clique in cliques:
        x = None
        for state in clique.states:
            if x is None:
                x = df.loc[pd.IndexSlice[state], :]['rtt']
            else:
                x = pd.concat([x, df.loc[pd.IndexSlice[state], :]['rtt']])

        color = colors[clique.cid % len(colors)]
        plt.hist(x, bins=400, density=True, cumulative=True, label="{}".format('%s' % ', '.join(clique.states)),
                 histtype='step', alpha=0.8, color=color, range=(0, 100))
        plt.axvline(x=np.median(x), color=color)
    plt.legend(loc='upper left')
    plt.savefig('{}/clusters_cdf.png'.format(results_dir))
    plt.show()


@total_ordering
class CliqueOfStates:
    def __init__(self, cid, states, ranks, values):
        self.cid = cid
        self.states = states
        self.ranks = ranks
        self.values = values

    def set_id(self, cid):
        self.cid = cid

    def mean_value(self):
        return np.mean(self.values)

    def __str__(self):
        state_list_string = ""
        for state, rank, value in sorted(zip(self.states, self.ranks, self.values), key=lambda x: x[1]):
            state_list_string += "\n\t{}\t{}\t{}".format(state, rank, value)

        return "Clique {}: {}\nMean value: {}".format(self.cid, state_list_string, self.mean_value())

    def __eq__(self, other):
        return self.mean_value() == other.mean_value()

    def __lt__(self, other):
        return self.mean_value() < other.mean_value()


def getDataFrame(column='rtt', z=2, cov_max=0.1):
    if not os.path.isdir("./pickles"):
        os.mkdir("pickles")

    pickle_filename = './pickles/{}_{}_{}_dataFrame_pickle.zip'.format(column, z, cov_max)

    if not os.path.isfile(pickle_filename):
        print("Creating pickle...")

        rawDf = pd.read_json('siteping_cleaned3_pretty.json')
        df = rawDf.loc[rawDf['isMobile'] == False]

        print("Filtering by RTT; RTT max/min/avg/stdev is {:.3f}/{:.3f}/{:.3f}/{:.3f}".format(df["rtt"].max(), df["rtt"].min(), df["rtt"].mean(), df["rtt"].std()))
        df = df[numpy.abs(stats.zscore(df["rtt"])) <= z]
        print("Filtered by RTT; RTT max/min/avg/stdev is now {:.3f}/{:.3f}/{:.3f}/{:.3f}".format(df["rtt"].max(), df["rtt"].min(), df["rtt"].mean(), df["rtt"].std()))

        # Determine the coefficient of variation for a given series
        def cov(x):
            return np.std(x) / np.mean(x)

        # Generate coefficient of variation for server pairs
        pairs = df.groupby('state')['rtt'].agg([np.median, cov])
        # Filter out pairs with coefficient of variations too high - these are unreliable
        pairs = pairs[pairs['cov'] < cov_max]

        del pairs['cov']

        pairs.to_pickle(pickle_filename)

    return pickle_filename


if __name__ == "__main__":
    main()