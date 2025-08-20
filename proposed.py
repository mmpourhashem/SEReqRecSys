import math
import os
# import string
import pandas as pd
import numpy as np
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import re
import csv
from collections import Counter
import helpers

class ProposedMethod:
    def __init__(self, rating, documents, sim_threshold, nn, method, woSemSim=False, woFlexW=False):
        self.method = method
        self.woSemSim = woSemSim
        self.woFlexW = woFlexW
        # self.fail_value = pd.to_numeric(rating.iloc[1:, 1:].stack(), errors='coerce').mean()
        self.fail_value = .5
        domainSpecificTFIDF = True
        if domainSpecificTFIDF:
            if os.path.isfile('DS2009Scores.csv'):
                self.scores = pd.read_csv('DS2009Scores.csv')
                min_score = self.scores['Score'].min()
                max_score = self.scores['Score'].max()
                self.scores['Normalized_Score'] = 1 - ((self.scores['Score'] - min_score) / (max_score - min_score))
            else:
                self.scores = self.domain_specific_2009()
                min_score = self.scores['Score'].min()
                max_score = self.scores['Score'].max()
                self.scores['Normalized_Score'] = 1 - ((self.scores['Score'] - min_score) / (max_score - min_score))
        else:
            self.scores = pd.read_csv('average_tfidf_scores.csv')
            min_score = self.scores['Score'].min()
            max_score = self.scores['Score'].max()
            self.scores['Normalized_Score'] = ((self.scores['Score'] - min_score) / (max_score - min_score))

        self.rating = rating
        self.documents = documents
        self.sim_threshold = sim_threshold
        self.sem_sim_threshold = .8
        self.defaultNeighborN = nn
        #uses glove similarity file, if does not exist, creates one.
        semSim_file_name = 'glove_sim_matrix' + str(method) + '.csv'
        if not os.path.isfile(semSim_file_name):
            self.glove_vectors = api.load("glove-wiki-gigaword-50")
            items = list(documents.keys())
            self.write_similarity_matrix_to_csv(items, semSim_file_name)
        self.gloveSim_df = pd.read_csv(semSim_file_name, index_col=0)

    def domain_specific_2009(self):

        # Function to compute Term Frequency (TF)
        # Formula: TF(token) = count_fg(token) / Σ_t∈fg count_fg(t)
        def compute_tf(token, domain_counts, fg_total):
            return domain_counts.get(token, 0) / fg_total

        # Function to compute Inverse Domain Frequency (IDF)
        # Formula: IDF(token) = log(|fg+bg| / (1 + |{d ∈ fg+bg : token ∈ d}|))
        def compute_idf(token, all_docs):
            total_doc_count = len(all_docs)

            # Count how many documents contain the token
            doc_count_with_token = sum(1 for doc in all_docs if token in doc.split())

            # IDF calculation
            return math.log(total_doc_count / (1 + doc_count_with_token))

        # Function to compute TF-IDF scores for each word
        # Formula: TF-IDF(token) = TF(token) * IDF(token)
        def compute_tfidf_scores(domain_counts, all_docs):
            fg_total = sum(domain_counts.values())

            # Compute TF-IDF for each word in domain corpus
            tfidf_scores = {}

            for word in domain_counts:
                tf = compute_tf(word, domain_counts, fg_total)
                idf = compute_idf(word, all_docs)
                tfidf_scores[word] = tf * idf  

            return tfidf_scores

        # Function to filter out non-alphabetic, short words
        def filter_words(tfidf_scores, min_word_length=3):
            filtered_scores = {word: score for word, score in tfidf_scores.items()
                            if re.match(r'^[a-zA-Z]+$', word) and len(word) >= min_word_length and word.lower()}
            return filtered_scores

        # Function to read text from a file with encoding handling
        def read_file(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()

        # Function to compute word counts from a document
        def get_word_counts(doc):
            return Counter(doc.split())

        # Read all files from a directory
        def read_directory(directory_path):
            all_docs = []
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path):
                    all_docs.append(read_file(file_path))
            return all_docs

        # Function to save scores to a CSV file
        def save_scores_to_csv(scores, output_file):
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Word', 'Score'])
                for word, score in scores.items():
                    writer.writerow([word, score])  

        domain_file_path = 'command.txt'
        background_directory =  os.path.join("processed-files", "")
        output_file_path = 'DS2009Scores.csv'

        # Read the domain-specific file
        domain_docs = read_file(domain_file_path).splitlines()

        # Read the background corpus from the directory of 79 files
        background_docs = read_directory(background_directory)

        # Get word counts for the domain (command) corpus
        domain_counts = get_word_counts(' '.join(domain_docs))
        all_docs = domain_docs + background_docs  # Combined foreground and background for IDF calculation

        # Calculate TF-IDF scores
        tfidf_scores = compute_tfidf_scores(domain_counts, all_docs)

        # Filter out non-alphabetic and short words and stop words
        filtered_scores = filter_words(tfidf_scores)

        # Print domain-specific words and their TF-IDF scores
        for word, score in tfidf_scores.items():
            print(f"Word: {word}, TF-IDF Score: {score}")

        # Save the filtered scores to a CSV file
        save_scores_to_csv(filtered_scores, output_file_path)

        # Return a DataFrame of the TF-IDF scores to use later
        df_scores = pd.DataFrame(filtered_scores.items(), columns=['Word', 'Score'])
        return df_scores
        
    def getColPredict(self, user, item):
        ratedItemsByUser = helpers.getUsersRatedItems(self.rating, user)
        similarities = []
        for item2 in ratedItemsByUser:
            sim = helpers.getHCBCFiSimCF(self.rating, item, item2)
            similarities.append((item2, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        topNItems = similarities[:self.defaultNeighborN]
        numerator = denominator = 0
        # counter = 0
        nSize = 0
        for item2, sim in topNItems:
            if sim >= self.sim_threshold:
                # counter += 1
                nSize += 1
                numerator += helpers.getRating(self.rating, user, item2) * sim
                denominator += abs(sim)
        
        # print('hybrid', counter)
        if denominator == 0:
            return (-2, nSize)
        return (numerator/denominator, nSize)

    def average_word_vectors(self, words, model, num_features):
        feature_vec = np.zeros((num_features,), dtype="float32")
        index2word_set = set(model.index_to_key)
        if isinstance(words, str):
            words = words.split()
        nwords = 0
        for word in words:
            if word in index2word_set:
                nwords += 1
                feature_vec = np.add(feature_vec, model[word])
        if nwords > 0:
            feature_vec = np.divide(feature_vec, nwords)
        return feature_vec

    def getGlove_sim1(self, item1, item2, num_features=50):
        if item1 not in self.documents or item2 not in self.documents:
            raise ValueError(f"One or both keys not found in documents: {item1}, {item2}")
        doc1 = self.documents[item1]
        doc2 = self.documents[item2]
        vec1 = self.average_word_vectors(doc1, self.glove_vectors, num_features)
        vec2 = self.average_word_vectors(doc2, self.glove_vectors, num_features)
        return cosine_similarity([vec1], [vec2])[0][0]

    def getGlove_sim2(self, item1, item2, num_features=50):
        if item1 not in self.documents or item2 not in self.documents:
            raise ValueError(f"One or both keys not found in documents: {item1}, {item2}")
        doc1 = self.documents[item1]
        doc2 = self.documents[item2]
        similarities = []
        weights = []
        for word1 in doc1.split():
            if word1 not in self.glove_vectors:
                continue
            for word2 in doc2.split():
                if word2 not in self.glove_vectors:
                    continue
                weight1 = self.get_importance_score(self.scores, word1)
                weight2 = self.get_importance_score(self.scores, word2)
                vec1 = self.glove_vectors[word1].reshape(1, -1)
                vec2 = self.glove_vectors[word2].reshape(1, -1)
                similarity = cosine_similarity(vec1, vec2)[0][0]
                weighted_similarity = similarity * weight1 * weight2
                similarities.append(weighted_similarity)
                weights.append(weight1 * weight2)
        if np.sum(weights) == 0:
            return 0
        average_similarity = np.sum(similarities) / np.sum(weights)
        print("req1:", doc1, "req2:", doc2, "sim:", average_similarity)
        return average_similarity

    def getGlove_sim3(self, item1, item2, num_features=50):
        if item1 not in self.documents or item2 not in self.documents:
            raise ValueError(f"One or both keys not found in documents: {item1}, {item2}")
        doc1 = self.documents[item1]
        doc2 = self.documents[item2]
        weighted_vec1 = np.zeros(num_features)
        weighted_vec2 = np.zeros(num_features)
        total_weight1 = 0
        total_weight2 = 0
        
        for word1 in doc1.split():
            if word1 in self.glove_vectors:
                weight1 = self.get_importance_score(self.scores, word1)
                weighted_vec1 += self.glove_vectors[word1] * weight1
                total_weight1 += weight1
        
        for word2 in doc2.split():
            if word2 in self.glove_vectors:
                weight2 = self.get_importance_score(self.scores, word2)
                weighted_vec2 += self.glove_vectors[word2] * weight2
                total_weight2 += weight2
        
        if total_weight1 == 0 or total_weight2 == 0:
            return 0
        weighted_vec1 /= total_weight1
        weighted_vec2 /= total_weight2
        similarity = cosine_similarity(weighted_vec1.reshape(1, -1), weighted_vec2.reshape(1, -1))[0][0]
        
        print("req1:", doc1, "req2:", doc2, "sim:", similarity)
        return similarity

    def getGlove_sim_from_file(self, item1, item2):
        return self.gloveSim_df.loc[item1, item2]
    
    def getProposedContentPredict(self, user, item):
        ratedItemsByUser = helpers.getUsersRatedItems(self.rating, user)
        similarities = []
        gSimilarities = []
        cSimilarities = []
        for item2 in ratedItemsByUser:
            cSimilarities.append((item2, self.getCategorySim(item, item2)))
            gSimilarities.append((item2, self.getGlove_sim_from_file(item, item2)))
        cSimilarities.sort(key=lambda x: x[1], reverse=True)
        gSimilarities.sort(key=lambda x: x[1], reverse=True)

        if self.woSemSim:
            similarities = cSimilarities[:self.defaultNeighborN]
        elif cSimilarities and cSimilarities[0][1] != 0:
            # Start with top-N category similarities
            similarities = cSimilarities[:self.defaultNeighborN]
            used_items = {item2 for item2, _ in similarities}
            # Replace entries with zero sim using gSimilarities if not already included
            for i in range(len(similarities)):
                if similarities[i][1] == 0:
                    for g_item2, g_sim in gSimilarities:
                        if g_item2 not in used_items and g_sim > self.sem_sim_threshold:
                            similarities[i] = (g_item2, g_sim)
                            used_items.add(g_item2)
                            break
        else:
            similarities = gSimilarities[:self.defaultNeighborN]

        top_n_similarities = similarities[:self.defaultNeighborN]
        numerator = denominator = 0
        for item2, sim in top_n_similarities:
            if sim > self.sim_threshold:
                numerator += helpers.getRating(self.rating, user, item2) * sim
                denominator += abs(sim)
        # if denominator == 0:
            # if self.woSemSim:
            #     print('Wos failed')
            # else:
            #     print('Method 2 failed')
        return numerator / denominator if denominator != 0 else self.fail_value #never happens with semantic similarity

    def getProposedPredict(self, user, item, lambda_param=0.5):
        # print(f"hybrid={self.getHybridPredict(df, user, item)}, content={self.getProposedContentPredict(df, user, item)}")
        (colPredict, neighborSize) = self.getColPredict(user, item)
        contentPredict = self.getProposedContentPredict(user, item)
        if colPredict == -2:
            if self.method == 2 and self.woFlexW:
                # colPredict = 0
                # print('Wof 2 failed')
                colPredict = self.fail_value
            if self.method == 2 and not self.woFlexW:
                return contentPredict
            if self.method == 1:
                # colPredict = 0
                # print('Method 1 failed')
                colPredict = self.fail_value
        newLambda = lambda_param * neighborSize / self.defaultNeighborN
        if self.method == 2 and not self.woFlexW:
            return (newLambda * colPredict) + (1 - newLambda) * contentPredict
        return (lambda_param * colPredict) + (1 - lambda_param) * contentPredict

    def create_item_similarity_matrix(self, items):
        num_items = len(items)
        sim_matrix = np.zeros((num_items, num_items))
        
        for i, item1 in enumerate(items):
            for j, item2 in enumerate(items):
                if i == j:
                    sim_matrix[i][j] = 1.0
                elif j > i:
                    if self.method == 1:
                        sim_matrix[i][j] = self.getGlove_sim1(item1, item2)
                    elif self.method == 2:
                        sim_matrix[i][j] = self.getGlove_sim2(item1, item2)
                    sim_matrix[j][i] = sim_matrix[i][j]
        
        return sim_matrix

    def normalize_matrix(self, matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        norm_matrix = (matrix - min_val) / (max_val - min_val)
        return norm_matrix

    def write_similarity_matrix_to_csv(self, items, filename):
        sim_matrix = self.create_item_similarity_matrix(items)
        norm_matrix = self.normalize_matrix(sim_matrix)
        df = pd.DataFrame(norm_matrix, index=items, columns=items)
        df.to_csv(filename)

    def getCategorySim(self,item1, item2):
        parts1 = item1.split('.')
        parts2 = item2.split('.')
        if parts1[0] == parts2[0]:
            if len(parts1) > 1 and len(parts2) > 1 and parts1[1] == parts2[1]:
                return 1
            else:
                return 0.5 #(0.66 best) #0.8
        else:
            return 0 #(0.33 best)
    
    def get_importance_score(self, df, word):
        result = df[df['Word'] == word]
        if not result.empty:
            return 1 + result['Normalized_Score'].values[0]
        else:
            return 1
