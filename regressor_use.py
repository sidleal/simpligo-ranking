import numpy as np
import sys
from keras.models import load_model

feats = ['sentence_length_max','words','words_per_sentence','brunet','ratio_coordinate_conjunctions','gunning_fox',
         'sentence_length_min','adjectives_min','punctuation_diversity','adjectives_max','dep_distance','flesch',
         'long_sentence_ratio','sentences_with_five_clauses','gerund_verbs','verbs','short_sentence_ratio','honore',
         'medium_long_sentence_ratio','yngve','coordinate_conjunctions_per_clauses','idade_aquisicao_1_25_ratio',
         'indicative_imperfect_ratio','concretude_mean','subjunctive_present_ratio','prepositions_per_sentence',
         'logic_operators','third_person_pronouns','relative_pronouns_ratio','ttr','aux_plus_PCP_per_sentence',
         'dalechall_adapted','tmp_pos_conn_ratio','ratio_subordinate_conjunctions','pronouns_max','pronoun_ratio',
         'tmp_neg_conn_ratio','sentences_with_six_clauses','log_pos_conn_ratio','abstract_nouns_ratio',
         'adverbs_ambiguity','frazier','apposition_per_clause','adjective_ratio','adjectives_ambiguity',
         'sentences_with_seven_more_clauses','sentences_with_four_clauses','subjunctive_imperfect_ratio',
         'imageabilidade_25_4_ratio','preposition_diversity','min_cw_freq','subordinate_clauses',
         'adverbs_diversity_ratio','idade_aquisicao_std','inflected_verbs','easy_conjunctions_ratio',
         'first_person_pronouns','familiaridade_4_55_ratio','if_ratio','familiaridade_mean','syllables_per_content_word',
         'postponed_subject_ratio','add_pos_conn_ratio','sentences_with_two_clauses','infinite_subordinate_clauses',
         'concretude_1_25_ratio','indicative_preterite_perfect_ratio','hypernyms_verbs','idade_aquisicao_mean',
         'max_noun_phrase','adverbs','concretude_std','nouns_ambiguity','idade_aquisicao_55_7_ratio','passive_ratio',
         'third_person_possessive_pronouns','oblique_pronouns_ratio','imageabilidade_55_7_ratio','verb_diversity',
         'subjunctive_future_ratio','simple_word_ratio','or_ratio','content_density','second_person_pronouns',
         'familiaridade_1_25_ratio','indefinite_pronoun_ratio','cau_pos_conn_ratio','relative_pronouns_diversity_ratio',
         'conn_ratio','add_neg_conn_ratio','first_person_possessive_pronouns','imageabilidade_std','indicative_present_ratio',
         'imageabilidade_mean','indicative_pluperfect_ratio','concretude_55_7_ratio','function_word_diversity','and_ratio',
         'pronoun_diversity','verbs_max','non-inflected_verbs','content_words','verbal_time_moods_diversity','personal_pronouns',
         'adverbs_before_main_verb_ratio','familiaridade_std','adverbs_min','adjunct_per_clause','medium_short_sentence_ratio',
         'infinitive_verbs','cau_neg_conn_ratio','sentences_with_zero_clause','adjective_diversity_ratio','content_word_diversity',
         'verbs_ambiguity','idade_aquisicao_25_4_ratio','nouns_min','log_neg_conn_ratio','cw_freq','nouns_max','adverbs_max',
         'familiaridade_25_4_ratio','sentences_with_three_clauses','named_entity_ratio_sentence','familiaridade_55_7_ratio',
         'content_word_min','relative_clauses','indefinite_pronouns_diversity','non_svo_ratio','imageabilidade_4_55_ratio',
         'ratio_function_to_content_words','clauses_per_sentence','temporal_adjunct_ratio','idade_aquisicao_4_55_ratio',
         'concretude_4_55_ratio','min_noun_phrase','words_before_main_verb','content_word_max','named_entity_ratio_text',
         'dialog_pronoun_ratio','punctuation_ratio','mean_noun_phrase','std_noun_phrase','function_words','pronouns_min',
         'negation_ratio','noun_diversity','verbs_min','prepositions_per_clause','participle_verbs','concretude_25_4_ratio',
         'indicative_condition_ratio','sentences_with_one_clause','noun_ratio','content_words_ambiguity','hard_conjunctions_ratio']

pss_min = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.7333, 0.0, 0.0, 0.1818, 0.0, 0.0, -259.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5455, 0.0, 0.0496, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1429, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1111, 0.0, 0.0, 0.0213, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
pss_max = np.array([70.0, 70.0, 70.0, 9.3501, 1.0, 28.1429, 70.0, 0.5, 1.0, 0.5, 250.0, 162.205, 1.0, 1.0, 1.0, 0.6667, 1.0, 6492.9972, 1.0, 7.1765, 4.0, 1.0, 1.0, 6.0985, 1.0, 17.0, 0.5, 1.0, 2.0, 1.0, 3.0, 19.5753, 0.5, 1.0, 0.5, 0.5, 0.0909, 1.0, 0.5, 0.5, 11.0, 14.0, 5.0, 0.5, 19.0, 1.0, 1.0, 1.0, 1.0, 1.0, 994512.0, 2.0, 1.0, 3.6141, 1.0, 0.5, 1.0, 1.0, 0.1429, 5.9759, 6.0, 1.0, 0.6667, 1.0, 1.0, 0.25, 1.0, 7.0, 7.7818, 56.0, 0.6, 1.6517, 20.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.2, 14.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.6667, 0.3333, 1.0, 1.2935, 1.0, 5.7876, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 5.0, 0.5, 2.0, 1.5827, 0.5, 6.0, 1.0, 1.0, 0.25, 1.0, 1.0, 1.0, 41.0, 1.0, 1.0, 0.25, 8720796.5455, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 6.0, 10.0, 3.0, 1.0, 1.0, 39.0, 44.0, 1.0, 0.5, 1.0, 1.5, 39.0, 27.5, 0.8571, 0.5, 0.5, 1.0, 0.5, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 0.3333])

pipeline_pss = load_model("models/model_regressor_v2_10.h5")

retStr = str(sys.argv[1])

retStr = retStr.replace("or_ratio ", "or_ratio")
retStr = retStr.replace("noun_ratio ", "noun_ratio")
ret = eval(retStr)

x = []
for f in feats:
    x.append(ret[f])

X = np.array(x)
X_norm = (X - pss_min) / (pss_max - pss_min)

prediction = pipeline_pss.predict(np.array([X_norm]))

result = round(prediction[0][0], 2)

if result < 0.01:
    result = 0.01
if result > 1:
    result = 1.0

print("result:",result)

