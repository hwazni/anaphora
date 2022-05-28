import pandas as pd
from tqdm import tqdm
from discopy import Box
from discopy.rigid import Spider
from discopro.grammar import tensor
from lambeq.core.types import AtomicType
from lambeq import BobcatParser, Rewriter
from discopro.rewriting import contract, pronoun_rule
from discopro.anaphora import connect_anaphora_on_top

S = AtomicType.SENTENCE
parser = BobcatParser()

rewriter = Rewriter(['auxiliary','connector','coordination','determiner','object_rel_pronoun',
                        'subject_rel_pronoun','postadverb','preadverb','prepositional_phrase'])

rewriter.add_rules(pronoun_rule)

join_box = Box('Box', S @ S, S)

def generate_diag(df, conf):

    diagrams = []

    for i in tqdm(range(df.shape[0])):

        s1 = df.iloc[i].sent1.strip().lower()
        s2 = df.iloc[i].sent2.strip().lower()

        pro = df.iloc[i].pronoun.strip().lower()
        ref = df.iloc[i].referent.strip().lower()

        try:
            
            d1 = parser.sentence2diagram(s1)
            d2 = parser.sentence2diagram(s2)

            diag = tensor(d1, d2)

            pro_box_idx = next(i for i, box in enumerate(diag.boxes) if box.name.casefold() == pro.casefold())
            ref_box_idx = next(i for i, box in enumerate(diag.boxes) if box.name.casefold() == ref.casefold())
            
            diag = connect_anaphora_on_top(diag, pro_box_idx, ref_box_idx)
            diag = rewriter(diag).normal_form()

            diag = contract(diag).normal_form()

            if conf == 'spider':
                diag = diag >> Spider(2, 1, S)
            
            if conf == 'box':
                diag = diag >> join_box
            
            diagrams.append(diag)

        except:
            print('error.')
            print(s1,s2)

    return diagrams