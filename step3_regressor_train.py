import numpy as np
import pandas
from keras import Model, layers
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split, KFold

shared_layer_1 = Dense(64, kernel_initializer='normal', activation='relu', name='shared_layer_1')
shared_layer_2 = Dense(100, kernel_initializer='normal', activation='relu', name='shared_layer_2')

def eye_model():
    input_layer_eye = Input(shape=(156,), name='eye_input')
    layer_1_eye = shared_layer_1(input_layer_eye)
    layer_2_eye = shared_layer_2(layer_1_eye)
    output_layer_eye = Dense(3, kernel_initializer='normal', activation="relu", name='eye_output')(layer_2_eye)
    model = Model(inputs=input_layer_eye, outputs=output_layer_eye)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def pss_model():
    input_layer_pss = Input(shape=(156,), name='pss_input')

    layer_1_pss = Dense(64, kernel_initializer='normal', activation='relu', name='layer_1_pss_1')(input_layer_pss)

    shared_layer_1_pss = shared_layer_1(input_layer_pss)
    shared_layer_1_pss.trainable = False

    shared_layer_2_pss = shared_layer_2(shared_layer_1_pss)
    shared_layer_2_pss.trainable = False

    eye_predict_pss = Dense(3, kernel_initializer='normal', activation="relu", name='eye_predict_pss_1')(shared_layer_2_pss)

    merged_pss = layers.concatenate([layer_1_pss, eye_predict_pss])

    output_layer_pss = Dense(1, kernel_initializer='normal', activation="relu", name='pss_output')(merged_pss)

    model = Model(inputs=input_layer_pss, outputs=output_layer_pss)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

pandas.set_option('display.max_colwidth', -1)

# load dataset eye
df_eye = pandas.read_csv("data/120sent_eye_features.tsv", delimiter='\t', header=0)

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
Y_eye = df_eye[['sum_first_pass', 'sum_regression', 'sum_total_pass']]

df_pss = pandas.read_csv("data/pss2_ranking_global_v3_raw.tsv", delimiter='\t', header=0)

X_pss = df_pss[feats]
Y_pss = df_pss['idx']

#normalize
X_eye = (X_eye-X_eye.min())/(X_eye.max()-X_eye.min())
Y_eye = (Y_eye-Y_eye.min())/(Y_eye.max()-Y_eye.min())
X_pss = (X_pss-X_pss.min())/(X_pss.max()-X_pss.min())
Y_pss = (Y_pss-Y_pss.min())/(Y_pss.max()-Y_pss.min())

pipeline_eye = eye_model()
pipeline_eye.fit(X_eye, Y_eye, epochs=100, batch_size=1, verbose=1)

pipeline_pss = pss_model()

total_err = 0
n_split = 10
count = 0
for train_index, test_index in KFold(n_splits=n_split, shuffle=True).split(X_pss):
    count+=1
    X_train_pss, Y_train_pss = X_pss.iloc[train_index], Y_pss.iloc[train_index]
    X_test_pss, Y_test_pss = X_pss.iloc[test_index], Y_pss.iloc[test_index]

    pipeline_pss.fit(X_train_pss, Y_train_pss, epochs=30, batch_size=10, verbose=0)

    prediction = pipeline_pss.predict(X_test_pss)
    y_test = np.asanyarray(Y_test_pss)

    error = 0
    for j in range(0, len(prediction)):
        #print("%s - %s" % (y_test[j], prediction[j]))
        error = error + (prediction[j] - y_test[j])**2

    print("MEAN SQUARED ERROR:", error/len(prediction))
    total_err += error/len(prediction)

    pipeline_pss.save("models/model_regressor_v2_%d.h5" % count)
    print("Saved model to disk: ", count)

print("MEAN SQUARED ERROR TOTAL:", total_err/n_split)

