window_data = {}
window_df_data = {}

for window_x in [0, 1]:
    print('Processing train ...')
    train_texts, data_corpus, train_labels, sentence_train, train_target_words, train_positions, a, train_pos_tags = read_lexical_corpus('data/raw/lcp_single_train.tsv',
                                        nlp=nlp,
                                        return_complete_sent=False,
                                        window_size=window_x + 1)

    print('Processing trial ...')
    trial_texts, trial_corpus, trial_labels, sentences_trial, trial_target_words, trial_positions, b, trial_pos_tags = read_lexical_corpus('data/raw/lcp_single_trial.tsv',
                                        nlp=nlp,
                                        return_complete_sent=False,
                                        window_size=window_x + 1)
    
    print('Processing test ...')
    test_texts, test_corpus, test_labels, sentences_test, test_target_words, test_positions, c, test_pos_tags = read_lexical_corpus('data/raw/lcp_single_test.tsv',
                                        nlp=nlp,
                                        return_complete_sent=False,
                                        window_size=window_x + 1)
    
    multi_texts, multi_corpus, multi_labels, sentence_multi, multi_target_words, multi_positions, d, multi_pos_tags = read_lexical_corpus('data/raw/lcp_multi_train.tsv',
                                        nlp=nlp,
                                        return_complete_sent=False,
                                        window_size=window_x + 1)

    window_data[window_x] = {'train': [], 'train_multi': [], 'val': [], 'test': []}
    window_data[window_x]['train'] = [train_texts, data_corpus, train_labels, sentence_train, train_target_words, train_positions, a, train_pos_tags]
    window_data[window_x]['val'] = [trial_texts, trial_corpus, trial_labels, sentences_trial, trial_target_words, trial_positions, b, trial_pos_tags]
    window_data[window_x]['test'] = [test_texts, test_corpus, test_labels, sentences_test, test_target_words, test_positions, c, test_pos_tags]
    
    window_data[window_x]['train_multi'] = [multi_texts, multi_corpus, multi_labels, sentence_multi, multi_target_words, multi_positions, d, multi_pos_tags]
    