'''
    CSCS 466 Project 2 : machine learning
    Makena Kong & Whitney Larsen

    Create Hearing File from Committee Utterances
'''
import pandas as pd

def main():
    file = '/lib/466/DigitalDemocracy/committee_utterances.tsv'
    data = pd.read_csv(file, sep="\t")
    hids = data.hid.unique()
    columns = ["hid","text"]
    hearing_data = pd.DataFrame(columns=columns)

    for id in hids:
        rows = data.loc[data.hid == id]
        new_text = rows.text.str.cat(sep=" ")
        #new_df = pd.DataFrame(columns=columns, data=[id,new_text])
        new_df = {'hid': id, 'text':new_text}
        hearing_data = hearing_data.append(new_df, ignore_index=True)

    #print(hearing_data)
    hearing_data.to_csv("committee_hearings.tsv", sep="\t")


if __name__=="__main__":
    main()
