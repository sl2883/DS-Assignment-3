import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

alex_index = 499
req_score = 5
to_predict_from = 100

shows = np.loadtxt('Assignment-2-data/shows.txt', usecols=range(1), dtype='str', delimiter='\n')
R = np.loadtxt('Assignment-2-data/user-shows.txt', usecols=range(shows.size))

R_alex = np.array([R[alex_index, :]])
cos_sim = cosine_similarity(R)

cos_sim_alex = cos_sim[alex_index, :]

S = cos_sim_alex @ R

unknown_ratings = S[:to_predict_from]

sorted_index_unknowns = np.argsort(unknown_ratings)[::-1]

for i in range(0, req_score):
    print(shows[sorted_index_unknowns[i]], unknown_ratings[sorted_index_unknowns[i]])

