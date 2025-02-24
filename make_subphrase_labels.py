import os
import json
import random
import natural_logic_model as nlm
import data_util as du
import generate_data as gd
import tqdm

data, _, _ = gd.process_data(1.0)
def adjoin(words):
    result = ''
    for word in words:
        result += word + " "
    return result[:-1]

prefixes = ["0", "0.0625", "0.125", "0.25", "0.375","0.5","0.625", "0.75"]
for prefix in prefixes:
    if not os.path.exists(prefix + "gendata.train"):
        continue
        
    with open( prefix + "gendata.train", "r") as f:
        examples = []
        labels = []
        with tqdm.tqdm(total=len(f.readlines())) as tbar:
            for line in f.readlines():
                tbar.update(1)
                example = json.loads(line[:-1])
                label = []
                for i in [1,2,4,5,7,8]:
                    example1=dict()
                    example1["sentence1"] = example["sentence1"].split()[i]
                    example1["sentence2"] =example["sentence2"].split()[i]
                    if example["sentence1"].split()[i] != "emptystring" and example["sentence2"].split()[i] == "emptystring":
                        example1["gold_label"] = "entails"
                    elif example["sentence1"].split()[i] == "emptystring" and example["sentence2"].split()[i] != "emptystring":
                        example1["gold_label"] = "reverse entails"
                    elif example["sentence2"].split()[i] == example["sentence1"].split()[i]:
                        example1["gold_label"] = "equivalence"
                    else:
                        example1["gold_label"] = "independence"
                    label.append(example1["gold_label"])
                for i in [1,4,7]:
                    example2=dict()
                    example2["sentence1"] = example["sentence1"].split()[i] + " " + example["sentence1"].split()[i+1]
                    example2["sentence2"] =example["sentence2"].split()[i]+ " " + example["sentence2"].split()[i+1]
                    if example["sentence2"].split()[i] == example["sentence1"].split()[i] and example["sentence2"].split()[i+1] == example["sentence1"].split()[i+1]:
                        example2["gold_label"] = "equivalence"
                    elif example["sentence2"].split()[i] == "emptystring" and example["sentence2"].split()[i+1] == example["sentence1"].split()[i+1]:
                        example2["gold_label"] = "entails"
                    elif example["sentence1"].split()[i] == "emptystring" and example["sentence2"].split()[i+1] == example["sentence1"].split()[i+1]:
                        example2["gold_label"] = "reverse entails"
                    else:
                        example2["gold_label"] = "independence"
                    label.append(example2["gold_label"])
                example5=dict()
                example5["sentence1"] = adjoin(example["sentence1"].split()[-5:])
                example5["sentence2"] =adjoin(example["sentence2"].split()[-5:])
                premise = du.parse_sentence(data, example["sentence1"])[0]
                hypothesis = du.parse_sentence(data, example["sentence2"])[0]
                verb_relation = nlm.standard_lexical_merge(premise.verb,hypothesis.verb)
                adverb_relation = nlm.standard_lexical_merge(premise.adverb,hypothesis.adverb)
                object_negation_signature = nlm.negation_merge(premise.object_negation, hypothesis.object_negation)
                object_determiner_signature = nlm.determiner_merge(premise.natlog_object_determiner, hypothesis.natlog_object_determiner)
                object_noun_relation = nlm.standard_lexical_merge(premise.object_noun,hypothesis.object_noun)
                object_adjective_relation = nlm.standard_lexical_merge(premise.object_adjective,hypothesis.object_adjective)
                VP_relation = nlm.standard_phrase(adverb_relation, verb_relation)
                object_NP_relation = nlm.standard_phrase(object_adjective_relation, object_noun_relation)
                object_DP_relation = nlm.determiner_phrase(object_determiner_signature, object_NP_relation, VP_relation)
                example5["gold_label"] =nlm.negation_phrase(object_negation_signature, object_DP_relation)
                if example5["gold_label"] in ["contradiction"]:
                    example5["gold_label"] += "2"
                label.append(example5["gold_label"])
                example6=dict()
                example6["sentence1"] = adjoin(example["sentence1"].split()[-6:])
                example6["sentence2"] =adjoin(example["sentence2"].split()[-6:])
                premise = du.parse_sentence(data, example["sentence1"])[0]
                hypothesis = du.parse_sentence(data, example["sentence2"])[0]
                verb_relation = nlm.standard_lexical_merge(premise.verb,hypothesis.verb)
                adverb_relation = nlm.standard_lexical_merge(premise.adverb,hypothesis.adverb)
                object_negation_signature = nlm.negation_merge(premise.object_negation, hypothesis.object_negation)
                object_determiner_signature = nlm.determiner_merge(premise.natlog_object_determiner, hypothesis.natlog_object_determiner)
                object_noun_relation = nlm.standard_lexical_merge(premise.object_noun,hypothesis.object_noun)
                object_adjective_relation = nlm.standard_lexical_merge(premise.object_adjective,hypothesis.object_adjective)
                verb_negation_signature = nlm.negation_merge(premise.verb_negation, hypothesis.verb_negation)
                VP_relation = nlm.standard_phrase(adverb_relation, verb_relation)
                object_NP_relation = nlm.standard_phrase(object_adjective_relation, object_noun_relation)
                object_DP_relation = nlm.determiner_phrase(object_determiner_signature, object_NP_relation, VP_relation)
                object_negDP_relation = nlm.negation_phrase(object_negation_signature, object_DP_relation)
                example6["gold_label"] = nlm.negation_phrase(verb_negation_signature, object_negDP_relation)
                if example6["gold_label"] in ["contradiction"]:
                    example6["gold_label"] += "2"
                label.append(example6["gold_label"])
                label.append(example["gold_label"])
                example["gold_label"] = label
                examples.append(example)

    random.shuffle(examples)
    with open(prefix + "gendata.train" + "1256", 'w') as f:
        for example in examples:
            f.write(json.dumps(example)+ "\n")
