from pathlib import Path
import pandas as pd
import pandarallel
import numpy as np
import torch
import csv
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk import ngrams

import string
import spacy
import syllables
from polyglot.text import Text, Word
from collections import Counter

def get_meta(sentence, ref, nlp, option):
    try:
        temp = []
        doc = nlp(sentence)
        for token in doc:
            if option == 'pos':
                temp.append(token.pos_)
            elif option == 'dep':
                temp.append(token.head.text)
            elif option == 'ent':
                temp.append(token.ent.end_char)
            else:
                temp.append(token.text)
        return np.array(temp)        
    except:
        return []
    
def read_lexical_corpus(split_dir, nlp=None, return_complete_sent=False, window_size=3):
    if 'tsv' in split_dir:
        data = pd.read_csv(split_dir, sep='\t', quoting=csv.QUOTE_NONE)
    elif 'xlsx' in split_dir:
        data = pd.read_excel(split_dir)
    
    data.rename(columns={'subcorpus': 'corpus'}, inplace=True)
    data.token.fillna('null', inplace=True)
    
    data['pos_label'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'pos'), axis=1)
    data['sentence_pre'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'text'), axis=1)
    
    texts = []
    pos_tag_targets = []
    pos_tag_sentences = []
    labels = []
    sentence_raw = []
    target_words = []
    corpus = []
    positions = []
    extra_features = []
    
    for ix, row in data.iterrows():
        try:
            if ' ' in row.token:
                ix = row.sentence_pre.index(row.token.split(' '))
                position = [ix, ix+1]
            else:
                position = [row.sentence_pre.index(row.token)]
        except:
            if ' ' in row.token:
                token_find = row.token.split(' ')[0]
            else:
                token_find = row.token
                
            for ix, w in enumerate(row.sentence_pre):
                if token_find in w:
                    if ' ' in row.token:
                        position = [ix, ix+1]
                    else:
                        position = [ix]
            for ix, w in enumerate(row.sentence_pre):
                if token_find in w:
                    if ' ' in row.token:
                        position = [ix, ix+1]
                    else:
                        position = [ix]
                    
        if return_complete_sent:
            texts.append(row.sentence_pre)
            pos_word=4
        else:
            lim_inf = position[0] - window_size + 1
            if lim_inf < 0:
                lim_inf = 0
            
            if ' ' in row.token:
                lim_sup = position[1] + window_size
            else:
                lim_sup = position[0]
            #sentence_pre = ' '.join(row.sentence_pre[lim_inf:(position+window_size)])
            #texts.append(sentence)
            pos_word = 3
            
            tokens = row.sentence.partition(row.token)
            sentence = ' '.join(tokens[0].split(' ')[-window_size:]) + tokens[1] + ' '.join(tokens[2].split(' ')[:window_size])
            texts.append(sentence)
            
        tags = row.pos_label[lim_inf:lim_sup] 
        pos_tag_targets.append(row.pos_label[position[0]])
        pos_tag_sentences.append(tags)
        positions.append(pos_word)
        #positions.append(len(tokens[0].split(' ')[-window_size:]))
        if 'complexity' in row:
            labels.append(row.complexity)
        else:
            labels.append(1)
        sentence_raw.append(row.sentence)
        target_words.append(row.token)
        corpus.append(row.corpus)

    return np.array(texts), np.array(corpus), np.array(labels), np.array(sentence_raw), np.array(target_words), np.array(positions), np.array(pos_tag_sentences), np.array(pos_tag_targets)

def read_extra_features(split_dir, normal_wiki_ngram_2=None, normal_wiki_ngram_3=None,
                        simple_wiki_ngram_2=None, simple_wiki_ngram_3=None,
                        cbt_corpus_ngram_2=None, cbt_corpus_ngram_3=None,
                        normal_wiki=None, simple_wiki=None, lexicon_dir=None, brown_dict=None, lang_8_corpus=None, tatoeba=None, cbt_corpus=None, nlp=None):
    if 'tsv' in split_dir:
        data = pd.read_csv(split_dir, sep='\t', quoting=csv.QUOTE_NONE)
    elif 'xlsx' in split_dir:
        data = pd.read_excel(split_dir)
    
    data.rename(columns={'subcorpus': 'corpus'}, inplace=True)
    data.token.fillna('null', inplace=True)
    
    print('Generating dependencies corpus')
    data['pos_label'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'pos'), axis=1)
    data['sentence_pre'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'text'), axis=1)
    #data['entities_sent'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'ent'), axis=1)
    data['dep_target'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'dep'), axis=1)
    
    extra_features = []
    
    len_lang_8_words = len(lang_8_corpus.split(' '))
    len_tatoeba = len(tatoeba.split(' '))
    len_cbt = len(cbt_corpus.split(' '))
    len_normal_wiki = len(normal_wiki.split(' '))
    len_simple_wiki = len(simple_wiki.split(' '))
    
    print('Generating auxiliar complexity')
    extra_lexicon = pd.read_csv(lexicon_dir, sep='\t', names=['token', 'complex_aux'])
    extra_lexicon['token_l'] = extra_lexicon['token'].str.lower()
    data['token_l'] = data['token'].str.lower()
    data_merged = pd.merge(data, extra_lexicon[['token_l', 'complex_aux']], on='token_l', how='left')
    
    def find_position(row):
        try:
            if ' ' in row.token:
                ix = row.sentence_pre.index(row.token.split(' '))
                return [ix, ix+1]
            else:
                return [row.sentence_pre.index(row.token)]
        except:
            if ' ' in row.token:
                token_find = row.token.split(' ')[0]
            else:
                token_find = row.token
                
            for ix, w in enumerate(row.sentence_pre):
                if token_find in w:
                    if ' ' in row.token:
                        return [ix, ix+1]
                    else:
                        return [ix]
            if ' ' in row.token:
                token_find = row.token_l.split(' ')[0]
            else:
                token_find = row.token_l
                    
            for ix, w in enumerate(row.sentence_pre):
                if token_find in w:
                    if ' ' in row.token_l:
                        return [ix, ix+1]
                    else:
                        return [ix]
    
    print('Counting ...')
    
    data_merged['position'] = data_merged.parallel_apply(lambda x: find_position(x), axis=1)
    data_merged['pos_tag'] = data_merged.apply(lambda x: x.pos_label[x.position][0], axis=1)
    #data_merged['entity'] = data_merged.apply(lambda x: x.entities_sent[x.position], axis=1)
    data_merged['len_sentence'] = data_merged.parallel_apply(lambda x: len(x.sentence), axis=1)
    
    data_merged['len_token'] = data_merged.parallel_apply(lambda x: len(x.token), axis=1)
    
    data_merged['count_senses'] = data_merged.apply(lambda x: sum([len(wn.synsets(w)) for w in x.token.split(' ')]), axis=1)
    data_merged['count_tags'] = data_merged.parallel_apply(lambda x: sum([len(brown_dict[w.lower()]) for w in x.token.split(' ')]), axis=1)
    data_merged['count_syllables'] = data_merged.parallel_apply(lambda x: syllables.estimate(x.token), axis=1)
    data_merged['count_morphemes'] = data_merged.parallel_apply(lambda x: sum([len(Word(w, language='en').morphemes) for w in x.token.split(' ')]), axis=1)
    
    print('Counting ...')
    data_merged['count_after'] = data_merged.parallel_apply(lambda x: len(x.sentence.partition(x.token)[2].split(' ')), axis=1)
    data_merged['count_before'] = data_merged.parallel_apply(lambda x: len(x.sentence.partition(x.token)[0].split(' ')), axis=1)
    
    def get_features_from_corpus(row):
        
        count_lang_8 = lang_8_corpus.count(row.token)
        count_tatoeba = tatoeba.count(row.token)
        count_cbt = cbt_corpus.count(row.token)
        count_normal_wiki = normal_wiki.count(row.token)
        count_simple_wiki = simple_wiki.count(row.token)

        return pd.Series([count_lang_8, 
                   count_lang_8 / len_lang_8_words,
                   count_tatoeba,
                   count_tatoeba / len_tatoeba,
                   count_cbt,
                   count_cbt / len_cbt,
                   count_normal_wiki,
                   count_normal_wiki / len_normal_wiki,
                   count_simple_wiki,
                   count_simple_wiki / len_simple_wiki])
    
    print('Generating features from corpus ...')
    data_merged[['count_lang_8',
                'freq_lang_8',
                'count_tatoeba',
                'freq_tatoeba',
                'count_cbt',
                'freq_cbt',
                'count_normal_wiki',
                'freq_normal_wiki',
                'count_single_wiki',
                'freq_single_wiki']] = data_merged.parallel_apply(lambda x: get_features_from_corpus(x), axis=1)
    
    data_merged['count_dep'] = data_merged.parallel_apply(lambda x: Counter(x.dep_target)[x.token], axis=1)
    data_merged['count_words'] = data_merged.parallel_apply(lambda x: x.token.count(' '), axis=1)

    def get_tags_features(row):
        lim_aux = row.position[0] - 8
        if len(row.position) > 1:
            lim_sup = row.position[1]
        else:
            lim_sup = row.position[0]
        
        sentence_pre = ' '.join(row.sentence_pre[(0 if lim_aux < 0 else lim_aux):(lim_sup+7)])
        tags_cut_c = Counter(row.pos_label[(0 if lim_aux < 0 else lim_aux):(lim_sup+5)])
        count_nouns = tags_cut_c['NOUN'] if 'NOUN' in tags_cut_c else 0
        count_verbs = tags_cut_c['VERB'] if 'VERB' in tags_cut_c else 0
        ratio = (count_nouns / count_verbs) if count_nouns != 0 and count_verbs != 0 else 0
        
        return pd.Series([ratio,
                          tags_cut_c['PROPN'] if 'PROPN' in tags_cut_c else 0,
                          count_nouns,
                          tags_cut_c['ADV'] if 'ADV' in tags_cut_c else 0,
                          count_verbs,
                          tags_cut_c['PART'] if 'PART' in tags_cut_c else 0])

    print('Generating tags features ...')
    data_merged[['ratio',
                'count_propn',
                'count_noun',
                'count_adv',
                'count_verb',
                'count_part']] = data_merged.parallel_apply(lambda x: get_tags_features(x), axis=1)

    def get_ngram_features(row):
        if len(row.position) > 1:
            pos_after = row.position[1]
        else:
            pos_after = row.position[0]
            
        pos_before = row.position[0]

        if pos_after + 1 < len(row.sentence_pre):
            tuple_after = (row.sentence_pre[pos_after], row.sentence_pre[pos_after+1])
        else:
            tuple_after = (row.sentence_pre[pos_after], '.')
            
        if pos_before - 1 >= 0:
            tuple_before = (row.sentence_pre[pos_before-1], row.sentence_pre[pos_before])
        else:
            tuple_before = ('.', row.sentence_pre[pos_before])
        
        aux_features = []
        aux_features.append(normal_wiki_ngram_2[tuple_after])
        aux_features.append(simple_wiki_ngram_2[tuple_after])
        aux_features.append(cbt_corpus_ngram_2[tuple_after])
        aux_features.append(normal_wiki_ngram_2[tuple_before])
        aux_features.append(simple_wiki_ngram_2[tuple_before])
        aux_features.append(cbt_corpus_ngram_2[tuple_before])

        aux_features.append(normal_wiki_ngram_3[tuple_after])
        aux_features.append(simple_wiki_ngram_3[tuple_after])
        aux_features.append(cbt_corpus_ngram_3[tuple_after])
        aux_features.append(normal_wiki_ngram_3[tuple_before])
        aux_features.append(simple_wiki_ngram_3[tuple_before])
        aux_features.append(cbt_corpus_ngram_3[tuple_before])

        return pd.Series(aux_features)
    
    print('Generating ngram features ...')
    data_merged[['count_ngram_2_simple_wiki_after',
                'count_ngram_2_normal_wiki_after',
                'count_ngram_2_cbt_corpus_after',
                'count_ngram_2_simple_wiki_before',
                'count_ngram_2_normal_wiki_before',
                'count_ngram_2_cbt_corpus_before',
                'count_ngram_3_simple_wiki_after',
                'count_ngram_3_normal_wiki_after',
                'count_ngram_3_cbt_corpus_after',
                'count_ngram_3_simple_wiki_before',
                'count_ngram_3_normal_wiki_before',
                'count_ngram_3_cbt_corpus_before']] = data_merged.apply(lambda x: get_ngram_features(x), axis=1)
    
    return data_merged.drop(['sentence', 'token', 'token_l'], axis=1)

def read_cwi_corpus(split_dir, nlp=None, return_complete_sent=False, window_size=3):
    if 'tsv' in split_dir:
        data = pd.read_csv(split_dir, sep='\t', quoting=csv.QUOTE_NONE, names=['id', 
                                                                              'sentence',
                                                                              'start',
                                                                              'end',
                                                                              'token',
                                                                              'n_ann',
                                                                              'n_not_ann',
                                                                              'n_ann_d',
                                                                              'n_not_ann_d',
                                                                              'bin',
                                                                              'complexity'])
    elif 'xlsx' in split_dir:
        data = pd.read_excel(split_dir)
    
    data.rename(columns={'subcorpus': 'corpus'}, inplace=True)
    data.token.fillna('null', inplace=True)
    
    data['pos_label'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'pos'), axis=1)
    data['sentence_pre'] = data.apply(lambda x: get_meta(x.sentence, x.token, nlp, 'text'), axis=1)
    
    texts = []
    pos_tag_targets = []
    labels = []
    sentence_raw = []
    target_words = []
    corpus = []
    positions = []
    extra_features = []
    
    for ix, row in data.iterrows():
        try:
            if ' ' in row.token:
                ix = row.sentence_pre.index(row.token.split(' ')[0])
                position = [ix, ix+row.token.count(' ')]
                print('hola')
            else:
                position = [row.sentence_pre.index(row.token)]
        except:
            if ' ' in row.token:
                token_find = row.token.split(' ')[0]
            else:
                token_find = row.token
            
            for ix, w in enumerate(row.sentence.split(' ')):
                if token_find in w:
                    if ' ' in row.token:
                        position = [ix, ix+row.token.count(' ')]
                    else:
                        position = [ix]
                    
        if False:
            texts.append(row.sentence_pre)
            pos_word=4
        else:
            lim_inf = position[0] - window_size + 1
            if lim_inf < 0:
                lim_inf = 0
            
            if ' ' in row.token:
                try:
                    lim_sup = position[1] + window_size
                except:
                    print(position)
                    print(row)
                    print(row.roken)
                    print(hola)
            else:
                lim_sup = position[0]
            #sentence_pre = ' '.join(row.sentence_pre[lim_inf:(position+window_size)])
            #texts.append(sentence)
            pos_word = 3
            
            tokens = row.sentence.partition(row.token)
            sentence = ' '.join(tokens[0].split(' ')[-window_size:]) + tokens[1] + ' '.join(tokens[2].split(' ')[:window_size])
            texts.append(sentence)
            
        tags = row.pos_label[lim_inf:lim_sup] 
        pos_tag_targets.append(row.pos_label[(position[0] if position[0] < len(row.pos_label) else (len(row.pos_label) - 1))])
        positions.append(pos_word)
        #positions.append(len(tokens[0].split(' ')[-window_size:]))
        if 'complexity' in row:
            labels.append(row.complexity)
        else:
            labels.append(1)

        target_words.append(row.token)

    return np.array(texts), np.array(labels), np.array(target_words), np.array(positions), np.array(pos_tag_targets)
