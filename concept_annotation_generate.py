import os
import json
from data_util import sentence
from natural_logic_model import negation_merge, determiner_merge, standard_lexical_merge, standard_phrase, determiner_phrase, negation_phrase
import tqdm 
def parse_simple_sentence(input_sentence):
    #Takes a simple input_sentence and outputs the corresponding
    #instance of the sentence class
    words = input_sentence.split()

    if words[0] == "notevery":
        subject_determiner = "not every"
        words = words[1:]
    else:
        subject_determiner = words[0]
        words = words[1:]

    if words[0] != 'emptystring':
        subject_adjective = words[0]
        words = words[1:]
    else:
        subject_adjective = ""
        words = words[1:]

    subject_noun = words[0]
    words = words[1:]

    if words[0] == "doesnot":
        negation = True
        words = words[1:]
    else:
        negation = False
        words = words[1:]

    if words[0] != 'emptystring':
        adverb = words[0]
        words = words[1:]
    else:
        adverb = ""
        words = words[1:]

    verb = words[0]
    words = words[1:]

    if words[0] == "notevery":
        object_determiner = "not every"
        words = words[1:]
    else:
        object_determiner = words[0]
        words = words[1:]

    if words[0] != 'emptystring':
        object_adjective = words[0]
        words = words[1:]
    else:
        object_adjective = ""
        words = words[1:]

    object_noun = words[0]

    return sentence(subject_noun, verb, object_noun, negation, adverb, subject_adjective, object_adjective, subject_determiner, object_determiner)

def construcut_h_and_q(sentence1, sentence2):
    premise = parse_simple_sentence(sentence1)
    hypothesis = parse_simple_sentence(sentence2)
    return premise,hypothesis

concept_label_list = {
    "qs":[0,1,2,3],
    "adjs":["equivalence","reverse entails","entails","independence"],
    "ns":["equivalence","reverse entails","entails","independence"],
    "neg":[0,1,2,3],
    "adv":["equivalence","reverse entails","entails","independence"],
    "v":["equivalence","reverse entails","entails","independence"],
    "qo":[0,1,2,3],
    "adjo":["equivalence","reverse entails","entails","independence"],
    "no":["equivalence","reverse entails","entails","independence"],
    "av":["equivalence","reverse entails","entails","independence"],
    "ans":["equivalence","reverse entails","entails","independence"],
    "ano":["equivalence","reverse entails","entails","independence"],
    "o":["equivalence", "entails", "reverse entails", "contradiction", "cover", "alternation", "independence"],
    "vo":["equivalence", "entails", "reverse entails", "contradiction", "cover", "alternation", "independence"]
}

def concept_annotation(premise,hypothesis):
    # Q_s
    subject_negation_signature, qs_option_id = negation_merge(premise.subject_negation, hypothesis.subject_negation, return_option=True)
    subject_determiner_signature = determiner_merge(premise.natlog_subject_determiner, hypothesis.natlog_subject_determiner)
    # N_s
    subject_noun_relation = standard_lexical_merge(premise.subject_noun,hypothesis.subject_noun)
    # Ajd_s
    subject_adjective_relation = standard_lexical_merge(premise.subject_adjective,hypothesis.subject_adjective)
    # Neg
    verb_negation_signature, neg_option_id = negation_merge(premise.verb_negation, hypothesis.verb_negation, return_option=True)
    # V
    verb_relation = standard_lexical_merge(premise.verb,hypothesis.verb)
    # Adv
    adverb_relation = standard_lexical_merge(premise.adverb,hypothesis.adverb)
    # Q_o
    object_negation_signature, qo_option_id = negation_merge(premise.object_negation, hypothesis.object_negation, return_option=True)
    object_determiner_signature = determiner_merge(premise.natlog_object_determiner, hypothesis.natlog_object_determiner)
    # N_o
    object_noun_relation = standard_lexical_merge(premise.object_noun,hypothesis.object_noun)
    # Ajd_o
    object_adjective_relation = standard_lexical_merge(premise.object_adjective,hypothesis.object_adjective)

    #the nodes of the tree
    # Adv + V (verb)
    VP_relation = standard_phrase(adverb_relation, verb_relation)
    # Ajd_o + N_o
    object_NP_relation = standard_phrase(object_adjective_relation, object_noun_relation)
    # Ajd_s + N_s
    subject_NP_relation = standard_phrase(subject_adjective_relation, subject_noun_relation)
    #  (Adv + V (verb)) + Q_o + (Ajd_o + N_o)
    object_DP_relation = determiner_phrase(object_determiner_signature, object_NP_relation, VP_relation)
    object_negDP_relation = negation_phrase(object_negation_signature, object_DP_relation)
    #  Neg + (Adv + V (verb)) + Q_o + (Ajd_o + N_o)
    negverb_relation = negation_phrase(verb_negation_signature, object_negDP_relation)

    concept_annotation = {
        "qs": concept_label_list["qs"].index(qs_option_id),
        "adjs": concept_label_list["adjs"].index(subject_adjective_relation),
        "ns": concept_label_list["ns"].index(subject_noun_relation),
        "neg": concept_label_list["neg"].index(neg_option_id),
        "adv": concept_label_list["adv"].index(adverb_relation),
        "v": concept_label_list["v"].index(verb_relation),
        "qo": concept_label_list["qo"].index(qo_option_id),
        "adjo": concept_label_list["adjo"].index(object_adjective_relation),
        "no": concept_label_list["no"].index(object_noun_relation),
        "av": concept_label_list["av"].index(VP_relation),
        "ans": concept_label_list["ans"].index(subject_NP_relation),
        "ano": concept_label_list["ano"].index(object_NP_relation),
        "o": concept_label_list["o"].index(object_negDP_relation),
        "vo": concept_label_list["vo"].index(negverb_relation)
    }

    return concept_annotation



if __name__ == "__main__":

    file_list = ["gendata.test","gendata.train","gendata.val"]

    dataset_path = "data/mqnli_causal"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    label_object = json.dumps(concept_label_list)
    with open(os.path.join(dataset_path,"mqnli_concept_label.json"),'w') as label_file:
        label_file.write(label_object)

    ratio_list = [0,0.0625, 0.125, 0.25, 0.5, 0.75]
    with tqdm.tqdm(total=len(ratio_list)*len(file_list)) as tbar:
        for ratio in ratio_list:
            for file in file_list:
                file_name = f"{ratio}{file}"
                dataset_file = open(file_name, 'r')
                new_dataset = []
                for sample in dataset_file.readlines():
                    sample_dict = json.loads(sample)
                    p,h = construcut_h_and_q(sample_dict["sentence1"],sample_dict["sentence2"])
                    concept_label = concept_annotation(p,h)
                    sample_dict["input"] = (sample_dict["sentence1"].replace("emptystring ","")+" .",sample_dict["sentence2"].replace("emptystring ","")+" .")
                    sample_dict["concept_label"] = concept_label
                    new_dataset.append(sample_dict)

                json_object = json.dumps(new_dataset)

                with open(os.path.join(dataset_path,file_name+".json"),'w') as output_file:
                    output_file.write(json_object)

                tbar.update(1)

