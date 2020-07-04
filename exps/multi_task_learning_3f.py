import numpy
import numpy as np
import pandas
from keras import Model, layers
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Input
from keras.wrappers.scikit_learn import KerasRegressor
import pickle
from sklearn.model_selection import KFold

input_size_pss = 312 #156 * 2
input_size_eye = 156

def baseline_model():
    # create model
    input_layer_pss = Input(shape=(input_size_pss,), name='pss_input')
    input_layer_eye = Input(shape=(input_size_eye,), name='eye_input')

    layer_1_pss = Dense(64, kernel_initializer='normal', activation='relu', name='pss_layer_1')(input_layer_pss)
    layer_1_eye = Dense(64, kernel_initializer='normal', activation='relu', name='eye_layer_1')(input_layer_eye)

    merged = layers.concatenate([layer_1_pss, layer_1_eye])

    shared_layer = Dense(100, activation='relu', name='shared_layer')(merged)

    output_layer_pss = Dense(1, kernel_initializer='normal', activation="sigmoid", name='pss_output')(shared_layer)
    output_layer_eye = Dense(3, kernel_initializer='normal', activation="relu", name='eye_output')(shared_layer)

    model = Model(inputs=[input_layer_pss, input_layer_eye], outputs=[output_layer_pss, output_layer_eye])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


pandas.set_option('display.max_colwidth', -1)

# load dataset eye
df_eye_tmp = pandas.read_csv("../data/120sent_eye_features.tsv", delimiter='\t', header=0)

#data augmentation
df_eye = pandas.DataFrame(numpy.repeat(df_eye_tmp.values, 50, axis=0))
df_eye.columns = df_eye_tmp.columns

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


X_eye = df_eye[feats]
#Y_eye = df_eye.iloc[:, 192] #avg_first_pass
#Y_eye = df_eye.iloc[:, 193] #avg_regression
#Y_eye = df_eye.iloc[:, 194] #avg_total_pass
#Y_eye = df_eye.iloc[:, 195] #tot_first_pass
#Y_eye = df_eye.iloc[:, 196] #tot_regression
#Y_eye = df_eye.iloc[:, 197] #tot_total_pass
Y_eye = df_eye[['sum_first_pass', 'sum_regression', 'sum_total_pass']]

# load dataset
df_pss = pandas.read_csv("../data/pss2_features_pairs_align2.tsv", delimiter='\t', header=0)

feats_from = ['fr_sentence_length_max','fr_words','fr_words_per_sentence','fr_brunet','fr_ratio_coordinate_conjunctions','fr_gunning_fox',
         'fr_sentence_length_min','fr_adjectives_min','fr_punctuation_diversity','fr_adjectives_max','fr_dep_distance','fr_flesch',
         'fr_long_sentence_ratio','fr_sentences_with_five_clauses','fr_gerund_verbs','fr_verbs','fr_short_sentence_ratio','fr_honore',
         'fr_medium_long_sentence_ratio','fr_yngve','fr_coordinate_conjunctions_per_clauses','fr_idade_aquisicao_1_25_ratio',
         'fr_indicative_imperfect_ratio','fr_concretude_mean','fr_subjunctive_present_ratio','fr_prepositions_per_sentence',
         'fr_logic_operators','fr_third_person_pronouns','fr_relative_pronouns_ratio','fr_ttr','fr_aux_plus_PCP_per_sentence',
         'fr_dalechall_adapted','fr_tmp_pos_conn_ratio','fr_ratio_subordinate_conjunctions','fr_pronouns_max','fr_pronoun_ratio',
         'fr_tmp_neg_conn_ratio','fr_sentences_with_six_clauses','fr_log_pos_conn_ratio','fr_abstract_nouns_ratio',
         'fr_adverbs_ambiguity','fr_frazier','fr_apposition_per_clause','fr_adjective_ratio','fr_adjectives_ambiguity',
         'fr_sentences_with_seven_more_clauses','fr_sentences_with_four_clauses','fr_subjunctive_imperfect_ratio',
         'fr_imageabilidade_25_4_ratio','fr_preposition_diversity','fr_min_cw_freq','fr_subordinate_clauses',
         'fr_adverbs_diversity_ratio','fr_idade_aquisicao_std','fr_inflected_verbs','fr_easy_conjunctions_ratio',
         'fr_first_person_pronouns','fr_familiaridade_4_55_ratio','fr_if_ratio','fr_familiaridade_mean','fr_syllables_per_content_word',
         'fr_postponed_subject_ratio','fr_add_pos_conn_ratio','fr_sentences_with_two_clauses','fr_infinite_subordinate_clauses',
         'fr_concretude_1_25_ratio','fr_indicative_preterite_perfect_ratio','fr_hypernyms_verbs','fr_idade_aquisicao_mean',
         'fr_max_noun_phrase','fr_adverbs','fr_concretude_std','fr_nouns_ambiguity','fr_idade_aquisicao_55_7_ratio','fr_passive_ratio',
         'fr_third_person_possessive_pronouns','fr_oblique_pronouns_ratio','fr_imageabilidade_55_7_ratio','fr_verb_diversity',
         'fr_subjunctive_future_ratio','fr_simple_word_ratio','fr_or_ratio','fr_content_density','fr_second_person_pronouns',
         'fr_familiaridade_1_25_ratio','fr_indefinite_pronoun_ratio','fr_cau_pos_conn_ratio','fr_relative_pronouns_diversity_ratio',
         'fr_conn_ratio','fr_add_neg_conn_ratio','fr_first_person_possessive_pronouns','fr_imageabilidade_std','fr_indicative_present_ratio',
         'fr_imageabilidade_mean','fr_indicative_pluperfect_ratio','fr_concretude_55_7_ratio','fr_function_word_diversity','fr_and_ratio',
         'fr_pronoun_diversity','fr_verbs_max','fr_non-inflected_verbs','fr_content_words','fr_verbal_time_moods_diversity','fr_personal_pronouns',
         'fr_adverbs_before_main_verb_ratio','fr_familiaridade_std','fr_adverbs_min','fr_adjunct_per_clause','fr_medium_short_sentence_ratio',
         'fr_infinitive_verbs','fr_cau_neg_conn_ratio','fr_sentences_with_zero_clause','fr_adjective_diversity_ratio','fr_content_word_diversity',
         'fr_verbs_ambiguity','fr_idade_aquisicao_25_4_ratio','fr_nouns_min','fr_log_neg_conn_ratio','fr_cw_freq','fr_nouns_max','fr_adverbs_max',
         'fr_familiaridade_25_4_ratio','fr_sentences_with_three_clauses','fr_named_entity_ratio_sentence','fr_familiaridade_55_7_ratio',
         'fr_content_word_min','fr_relative_clauses','fr_indefinite_pronouns_diversity','fr_non_svo_ratio','fr_imageabilidade_4_55_ratio',
         'fr_ratio_function_to_content_words','fr_clauses_per_sentence','fr_temporal_adjunct_ratio','fr_idade_aquisicao_4_55_ratio',
         'fr_concretude_4_55_ratio','fr_min_noun_phrase','fr_words_before_main_verb','fr_content_word_max','fr_named_entity_ratio_text',
         'fr_dialog_pronoun_ratio','fr_punctuation_ratio','fr_mean_noun_phrase','fr_std_noun_phrase','fr_function_words','fr_pronouns_min',
         'fr_negation_ratio','fr_noun_diversity','fr_verbs_min','fr_prepositions_per_clause','fr_participle_verbs','fr_concretude_25_4_ratio',
         'fr_indicative_condition_ratio','fr_sentences_with_one_clause','fr_noun_ratio','fr_content_words_ambiguity','fr_hard_conjunctions_ratio']

feats_to = ['to_sentence_length_max','to_words','to_words_per_sentence','to_brunet','to_ratio_coordinate_conjunctions','to_gunning_fox',
         'to_sentence_length_min','to_adjectives_min','to_punctuation_diversity','to_adjectives_max','to_dep_distance','to_flesch',
         'to_long_sentence_ratio','to_sentences_with_five_clauses','to_gerund_verbs','to_verbs','to_short_sentence_ratio','to_honore',
         'to_medium_long_sentence_ratio','to_yngve','to_coordinate_conjunctions_per_clauses','to_idade_aquisicao_1_25_ratio',
         'to_indicative_imperfect_ratio','to_concretude_mean','to_subjunctive_present_ratio','to_prepositions_per_sentence',
         'to_logic_operators','to_third_person_pronouns','to_relative_pronouns_ratio','to_ttr','to_aux_plus_PCP_per_sentence',
         'to_dalechall_adapted','to_tmp_pos_conn_ratio','to_ratio_subordinate_conjunctions','to_pronouns_max','to_pronoun_ratio',
         'to_tmp_neg_conn_ratio','to_sentences_with_six_clauses','to_log_pos_conn_ratio','to_abstract_nouns_ratio',
         'to_adverbs_ambiguity','to_frazier','to_apposition_per_clause','to_adjective_ratio','to_adjectives_ambiguity',
         'to_sentences_with_seven_more_clauses','to_sentences_with_four_clauses','to_subjunctive_imperfect_ratio',
         'to_imageabilidade_25_4_ratio','to_preposition_diversity','to_min_cw_freq','to_subordinate_clauses',
         'to_adverbs_diversity_ratio','to_idade_aquisicao_std','to_inflected_verbs','to_easy_conjunctions_ratio',
         'to_first_person_pronouns','to_familiaridade_4_55_ratio','to_if_ratio','to_familiaridade_mean','to_syllables_per_content_word',
         'to_postponed_subject_ratio','to_add_pos_conn_ratio','to_sentences_with_two_clauses','to_infinite_subordinate_clauses',
         'to_concretude_1_25_ratio','to_indicative_preterite_perfect_ratio','to_hypernyms_verbs','to_idade_aquisicao_mean',
         'to_max_noun_phrase','to_adverbs','to_concretude_std','to_nouns_ambiguity','to_idade_aquisicao_55_7_ratio','to_passive_ratio',
         'to_third_person_possessive_pronouns','to_oblique_pronouns_ratio','to_imageabilidade_55_7_ratio','to_verb_diversity',
         'to_subjunctive_future_ratio','to_simple_word_ratio','to_or_ratio','to_content_density','to_second_person_pronouns',
         'to_familiaridade_1_25_ratio','to_indefinite_pronoun_ratio','to_cau_pos_conn_ratio','to_relative_pronouns_diversity_ratio',
         'to_conn_ratio','to_add_neg_conn_ratio','to_first_person_possessive_pronouns','to_imageabilidade_std','to_indicative_present_ratio',
         'to_imageabilidade_mean','to_indicative_pluperfect_ratio','to_concretude_55_7_ratio','to_function_word_diversity','to_and_ratio',
         'to_pronoun_diversity','to_verbs_max','to_non-inflected_verbs','to_content_words','to_verbal_time_moods_diversity','to_personal_pronouns',
         'to_adverbs_before_main_verb_ratio','to_familiaridade_std','to_adverbs_min','to_adjunct_per_clause','to_medium_short_sentence_ratio',
         'to_infinitive_verbs','to_cau_neg_conn_ratio','to_sentences_with_zero_clause','to_adjective_diversity_ratio','to_content_word_diversity',
         'to_verbs_ambiguity','to_idade_aquisicao_25_4_ratio','to_nouns_min','to_log_neg_conn_ratio','to_cw_freq','to_nouns_max','to_adverbs_max',
         'to_familiaridade_25_4_ratio','to_sentences_with_three_clauses','to_named_entity_ratio_sentence','to_familiaridade_55_7_ratio',
         'to_content_word_min','to_relative_clauses','to_indefinite_pronouns_diversity','to_non_svo_ratio','to_imageabilidade_4_55_ratio',
         'to_ratio_function_to_content_words','to_clauses_per_sentence','to_temporal_adjunct_ratio','to_idade_aquisicao_4_55_ratio',
         'to_concretude_4_55_ratio','to_min_noun_phrase','to_words_before_main_verb','to_content_word_max','to_named_entity_ratio_text',
         'to_dialog_pronoun_ratio','to_punctuation_ratio','to_mean_noun_phrase','to_std_noun_phrase','to_function_words','to_pronouns_min',
         'to_negation_ratio','to_noun_diversity','to_verbs_min','to_prepositions_per_clause','to_participle_verbs','to_concretude_25_4_ratio',
         'to_indicative_condition_ratio','to_sentences_with_one_clause','to_noun_ratio','to_content_words_ambiguity','to_hard_conjunctions_ratio']

all_feats = feats_from + feats_to
X_pss = df_pss[all_feats]
#Y_pss = df_pss.iloc[:, 3]
Y_pss = df_pss['inverse']

pipeline = baseline_model()

total_acuracia = 0
total_fscore = 0

#normalize
X_pss = (X_pss-X_pss.min())/(X_pss.max()-X_pss.min())
X_eye = (X_eye-X_eye.min())/(X_eye.max()-X_eye.min())
Y_eye = (Y_eye-Y_eye.min())/(Y_eye.max()-Y_eye.min())

X_eye_balanced = X_eye[:X_pss.shape[0]]
Y_eye_balanced = Y_eye[:Y_pss.shape[0]]

n_split = 10
for train_index, test_index in KFold(n_split).split(X_pss):
    X_train_pss, Y_train_pss = X_pss.iloc[train_index], Y_pss.iloc[train_index]
    X_test_pss, Y_test_pss = X_pss.iloc[test_index], Y_pss.iloc[test_index]
    X_train_eye, Y_train_eye = X_eye_balanced.iloc[train_index], Y_eye_balanced.iloc[train_index]
    #X_test_eye, Y_test_eye = X_eye_balanced.iloc[test_index], Y_eye_balanced.iloc[test_index]

    pipeline.fit([X_train_pss, X_train_eye], [Y_train_pss, Y_train_eye], epochs=30, batch_size=10, verbose=0)

    X_test_eye = np.random.randn(X_test_pss.shape[0], 156)

    prediction_tmp = pipeline.predict([X_test_pss, X_test_eye])

    prediction = prediction_tmp[0]
    y_test = numpy.asanyarray(Y_test_pss)

    for j in range(0, len(prediction)):
        if prediction[j] > 0.5:
            prediction[j] = 1
        else:
            prediction[j] = 0

    print(accuracy_score(y_test, prediction))

    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for j in range(0, len(prediction)):
        if y_test[j] == 1 and prediction[j] == 1:
            truePos+=1
        if y_test[j] == 0 and prediction[j] == 0:
            trueNeg+=1
        if y_test[j] == 1 and prediction[j] == 0:
            falseNeg+=1
        if y_test[j] == 0 and prediction[j] == 1:
            falsePos+=1

    acuracia = (truePos + trueNeg) / (truePos+trueNeg+falsePos+falseNeg)
    print("acuracia %s" % acuracia)
    total_acuracia += acuracia

    recall = truePos / (truePos+falseNeg)
    precision = truePos / (truePos+falsePos)
    specificity = trueNeg / (trueNeg+falsePos)

    fscore = (2 * (precision * recall)) / (precision + recall)

    print("---------------")
    print("recall %s" % recall)
    print("precision %s" % precision)
    print("specificity %s" % specificity)
    print("fscore %s" %fscore)
    total_fscore+= fscore
    print("---------------")

print("=================")
print("total acuracia %s" % (total_acuracia/n_split))
print("total fscore %s" % (total_fscore/n_split))
print("=================")


