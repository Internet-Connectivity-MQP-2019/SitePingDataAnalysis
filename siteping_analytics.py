from functools import total_ordering

import networkx
import pandas as pd
import numpy as np
from scipy.stats import kruskal, stats
from sklearn.cluster import AgglomerativeClustering
import os
import us
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    raw_df = pd.read_json('siteping_cleaned3_pretty.json')
    df = raw_df.loc[~raw_df['isMobile']]

    df = df.loc[df['state'].isin(list(us.states.mapping('abbr', 'name').keys()))]

    # Create main results directory if it doesn't exist
    if not os.path.isdir('./results'):
        os.mkdir('./results')

    results_dir = './results'

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

    # # Generate clusters based on inverted kruskal's values
    # ac = AgglomerativeClustering(n_clusters=None, compute_full_tree=True).fit(
    #     state_to_state_p_values_i)
    # state_rankings['cluster'] = ac.labels_

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
    net = networkx.from_pandas_adjacency(pValueDf)

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
            x = df[df['state'] == state]['rtt']

            color = colors[j % len(colors)]
            plt.hist(x, bins=400, normed=True, cumulative=True, label="{} ({:n})".format(state, clique.ranks[j]),
                     histtype='step', alpha=0.8, color=color, range=(0, 300))
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
                x = df[df['state'] == state]['rtt']
            else:
                x = pd.concat([x, df[df['state'] == state]['rtt']])

        color = colors[clique.cid % len(colors)]
        plt.hist(x, bins=400, normed=True, cumulative=True, label="{}".format('%s' % ', '.join(clique.states)),
                 histtype='step', alpha=0.8, color=color, range=(0, 300))
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


if __name__ == "__main__":
    main()
