import ijson
import pandas
import json
import numpy
from scipy.stats import stats

class StateData:
    def __init__(self, stateName):
        self.stateName = stateName
        self.data = []

    def addDatePoint(self, rtt):
        self.data.append(rtt)


def repairFile(filename):
    fp = open(filename, 'r')
    newfp = open("siteping_cleaned2.json", 'w')
    line = fp.readline()
    seenQuote = False
    seenBracket = False
    lineCount = 0
    while line:
        newLine = ""
        for char in line:
            if not seenQuote:
                if char == "\"":
                    seenQuote = True
                else:
                    newLine += char

            elif char != '\\':
                newLine += char
                if len(newLine) > 3:
                    threeChars = newLine[-3]+newLine[-2]+newLine[-1]
                    if threeChars == "\",\"":
                        newLine=newLine[0: -3]
                        newLine += ","
                        newfp.write(newLine+"\n")
                        lineCount += 1
                        newLine = ""
        newfp.write(newLine[0:-1])
        print(newLine)
        print(lineCount)
        line = fp.readline()
    fp.close()
    newfp.close()


def readFile (filename, zscore):
    releventColumns = [
        "favicon",
        "rtt",
        "state"
    ]

    releventColumnIndexs = [
        2,
        3,
        9
    ]
    with open(filename) as fp:
        file = json.load(fp)
        print(file[0]['favicon'])
        df = pandas.DataFrame(file)

    print("Filtering by RTT; RTT max/min/avg/stdev is {:.3f}/{:.3f}/{:.3f}/{:.3f}".format(df["rtt"].max(), df["rtt"].min(), df["rtt"].mean(), df["rtt"].std()))
    preFilterStateCounts = df['state'].value_counts()
    df = df[numpy.abs(stats.zscore(df["rtt"])) <= zscore]
    print("Filtered by RTT; RTT max/min/avg/stdev is now {:.3f}/{:.3f}/{:.3f}/{:.3f}".format(df["rtt"].max(), df["rtt"].min(), df["rtt"].mean(), df["rtt"].std()))
    postFilterStateCounts = df['state'].value_counts()
    df.set_index('state')

    stateData =  ((df.loc[df['isMobile'] == False]).groupby(['state']).rtt.agg(["min", "max", "mean", "median"]))
    stateData = stateData.merge(preFilterStateCounts.to_frame(), left_index=True, right_index=True)
    stateData = stateData.merge(postFilterStateCounts.to_frame(), left_index=True, right_index=True)
    stateData = stateData.rename(columns = {"state_x":"preFilterStateCounts", "state_y":"postFilterStateCounts" })
    stateData = stateData.merge(preFilterStateCounts.sub(postFilterStateCounts).to_frame(), left_index=True, right_index=True)
    stateData = stateData.rename(columns={"state": "difference"})

    stateData.to_csv("state_data.csv")












#repairFile("siteping_results.json")

readFile("siteping_cleaned3_pretty.json", 2)