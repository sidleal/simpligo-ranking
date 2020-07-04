import numpy
import pandas
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import pickle

input_size = 156

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation="relu"))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

pandas.set_option('display.max_colwidth', -1)

# load dataset
df = pandas.read_csv("../data/120sent_eye_features.tsv", delimiter='\t', header=0)

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

X = df[feats]
#X = df.iloc[:, 3:192]
#Y_tmp = df.iloc[:, 192] #avg_first_pass
#Y_tmp = df.iloc[:, 193] #avg_regression
#Y_tmp = df.iloc[:, 194] #avg_total_pass
#Y_tmp = df.iloc[:, 195] #tot_first_pass
#Y_tmp = df.iloc[:, 196] #tot_regression
#Y_tmp = df.iloc[:, 197] #tot_total_pass



Y = df[['avg_first_pass', 'avg_regression', 'avg_total_pass']]
#Y = df[['sum_first_pass', 'sum_regression', 'sum_total_pass']]

Y = (Y-Y.min())/(Y.max()-Y.min())

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=30, batch_size=1, verbose=1)

estimators = []
standardScaler = StandardScaler()
estimators.append(('standardize', standardScaler))
estimators.append(('mlp', estimator))
pipeline = Pipeline(estimators)

total_mse = 0
total_rmse = 0
total_r = 0
total_p = 0

folds = 10
for z in range (0, folds):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    pipeline.fit(X_train, y_train)
    prediction_tmp = pipeline.predict(X_test)

    y_test_tmp = numpy.asanyarray(y_test)

    #print(X_test)
    #print(y_test_tmp)
    #print(prediction_tmp)
    for i in range(0, 3):
        y_test = y_test_tmp[:, i]
        prediction = prediction_tmp[:, i]
        r, p_value = scipy.stats.pearsonr(y_test, prediction)
        print(i, "PEARSON'S CORRELATION COEFFICIENT:", r, "p-value", p_value)

        if numpy.math.isnan(r):
            r = 0

        error = 0
        tot_samples = len(prediction)
        for j in range(0, tot_samples ):
            #print("%s - %s" % (y_test[j], prediction[j]))
            error = error + (prediction[j] - y_test[j]) ** 2

        mse = error / len(prediction)
        rmse = numpy.math.sqrt(mse)
        print(i, "MEAN SQUARED ERROR:", mse)
        print(i, "ROOT MEAN SQUARED ERROR:", rmse)

        total_r += r
        total_p += p_value
        total_mse += mse
        total_rmse += rmse
        print("---------------")

print("=================")
print("total mse %s" % (total_mse/(folds * 3)))
print("total rmse %s" % (total_rmse/(folds * 3)))
print("total person's %s - p-value %s" % (total_r/(folds*3), total_p/(folds * 3)))

print("=================")
