import ast

import numpy as np
import copy

debug = False


def generate_uniform_simplex(no_entries):
    ''' Generates a vector uniformly random in the (no_entries - 1) simplex.

    input: no_entries - number of entries in the vector
    -----
    output: w - numpy array in the (no_entries - 1) simplex.
    '''

    # Get an increasingly sorted list of (no_entries - 1) elements --> dividers of [0, 1]
    dividers = np.random.random(no_entries - 1)
    dividers = np.append(dividers, [0, 1])
    dividers.sort()

    # get the vector in the simplex (weights) given the dividers
    w = np.zeros(no_entries)
    for i in range(no_entries):
        w[i] = dividers[i + 1] - dividers[i]

    return w


class Attributes():
    def __init__(self, config, for_user=False):
        self.config = config

        # True iff these are user atributes (--> all competing values are 1)
        self.for_user = for_user

        self.values = self.generate_values()

    def generate_values(self):
        '''Generates matching and competing attributes for items, and matching attributes for users.
        ---
        ToDo: also make versions for idiosyncratic taste
        '''

        num_attributes = self.config['num_attributes']
        kind_attributes_exp = self.config['kind_attributes_expanded']
        bound = self.config['matching_bound']

        total_num_attributes = sum(num_attributes)
        mean = np.zeros(total_num_attributes)
        cov = self.config['covariance']

        # generate the unrounded values
        attr = np.random.multivariate_normal(mean, cov)

        # conversion from normal attributes to 0/1 for competing and -1/+1 for matching
        def conversion(i):
            '''Conversion from normal attributes to 0/1 for competing and -1/+1 for matching.
            i = position of attribute --> type of attribute, and its value'''
            if 'm' in kind_attributes_exp[i]:
                return 1 if attr[i] > bound else -1
            else:
                if self.for_user:
                    return 1
                else:
                    # return 1 if attr[i] > 0 else 0
                    return attr[i]

        # print(attr)
        attr = [conversion(i) for i in range(total_num_attributes)]
        # print(attr)

        return attr

    def get_maching_attributes(self):
        '''Returns a list of the maching attributes of the CC.'''

        maching = []
        kind_attributes_exp = self.config['kind_attributes_expanded']

        for i, k in enumerate(kind_attributes_exp):
            if 'm' in k:
                maching.append(self.values[i])

        return maching


class User:
    def __init__(self, config, id, list_CCs, attributes=None, weights=None):
        self.config = config
        self.id = id

        # the attributes of the user (the positions for competing attributes are always 1)
        self.attributes = attributes
        if attributes is None:
            self.attributes = Attributes(config, True)

        # the importance the suer gives to each attribute (sums to 1)
        self.weights = weights
        if weights is None:
            self.weights = self.generate_weights()

        # the best CC followed so far
        self.best_followed_CC = None

        # best CCs according to the user (most likely a list of only one)
        self.ranking_CCs = self.get_ranking_CCs(list_CCs)

        # the user is protected if they have a maching attribute with value -1
        self.maching_attr = self.attributes.get_maching_attributes()
        self.protected = -1 in self.maching_attr

    def score(self, c):
        '''Evaluates the score the user associates with content creator c.

        input: c - a content creator
        -----
        ouptut: s - a value the user associates with c
        '''

        kind_attributes_exp = self.config['kind_attributes_expanded']

        # competing attributes add the quality of the CC + the imprtance of the attribute
        # matching attributes boost (reduce) the score by the weight if they (don't) match
        score = 0
        for i, k in enumerate(kind_attributes_exp):
            if 'c' in k:
                score += c.attributes.values[i] * self.weights[i]
            else:
                if self.attributes.values[i] == c.attributes.values[i]:
                    score += self.weights[i]
                else:
                    score -= self.weights[i]

        return score

    def decide_follow(self, c):
        '''Evaluates whether the user wants to follow CC c.

        input: c - a content creator
        ------
        output: bool - decision if it follows c'''

        score = self.score

        # it follows c iff they are better then the best followed so far
        if (self.best_followed_CC is None) or (score(self.best_followed_CC) < score(c)):
            self.best_followed_CC = c
            return True

        return False

    def get_ranking_CCs(self, list_CCs):
        '''Returns a dictionary of the CC and their position in the preference ranking or the user.
        If two are equally on the second position each of them maps to position 2.
        return: dict = {c.id: position_of_c_in_preference_of_u, ...}'''

        sorted_CCs = sorted(
            list_CCs, key=lambda c: self.score(c), reverse=True)
        scores = [self.score(c) for c in list_CCs]

        ranking = {}
        position = -1
        for p, c in enumerate(sorted_CCs):
            # we only increase the position when encountering a new score
            if p == 0 or scores[p - 1] != scores[p]:
                position += 1

            ranking[c.id] = position

        return ranking

    def generate_weights(self):
        '''Generates a weight vector --> the weights of a user for each attribute.
        num_attributes = [#entries for attributes of type 0, ...]
        cumulative_weights = [sum weights of attributes of type 0, ...]
        '''

        num_attributes = self.config['num_attributes']
        cumulative_weights_list = self.config['cumulative_weights']
        prob_cumulative_weights = self.config['prob_cumulative_weights']

        # generate a cumulative weight profile based on the given probabilities in config
        num_w = len(cumulative_weights_list)
        index_w = np.random.choice(range(num_w), p=prob_cumulative_weights)
        cumulative_weights = cumulative_weights_list[index_w]

        # if the we don't have constraints on the cumulative weights, then just get random weights
        if cumulative_weights == -1:
            return generate_uniform_simplex(sum(num_attributes))

        no_types = len(num_attributes)
        weights_parts = []

        for t in range(no_types):
            weights_parts.append(generate_uniform_simplex(
                num_attributes[t]) * cumulative_weights[t])

        weights = np.concatenate(weights_parts)

        return weights


# class CC:
#     def __init__(self, config, id, attributes=None):
#         self.config = config
#         self.id = id
#
#         # the attributes of the user (the positions for competing attributes are always 1)
#         self.attributes = attributes
#         if attributes is None:
#             self.attributes = Attributes(config)
#
#         # the content creator is protected if they have a maching attribute with value -1
#         self.maching_attr = self.attributes.get_maching_attributes()
#         self.protected = -1 in self.maching_attr
#
#     def get_competing_score(self):
#         '''Computes the sum of all competing attributes of the CC.'''
#
#         kind_attributes_exp = self.config['kind_attributes_expanded']
#
#         quality = 0
#         for i, k in enumerate(kind_attributes_exp):
#             if 'c' in k:
#                 quality += self.attributes.values[i]
#
#         return quality
#
#     def weight_followers_RS(self):
#         '''The RS could add biases to content crators.'''
#
#         if self.protected:
#             return 1 - self.config['level_bias_RS']
#         return 1


class CC:
    def __init__(self, config, id, attributes=None):
        # Basic initialization
        self.config = config
        self.id = id
        self.attributes = attributes if attributes is not None else Attributes(config)

        # Protected status (existing logic)
        self.maching_attr = self.attributes.get_maching_attributes()
        self.protected = -1 in self.maching_attr

        # Dynamic quality fields
        self.base_quality = self.get_competing_score()
        self.current_quality = self.base_quality
        self.quality_history = [self.base_quality]

        # Activity tracking
        self.active = True
        self.posts_recent = 0
        self.posts_history = []
        self.followers_in_current_step = 0
        self.total_followers_count = 0
        self.active_time = 0
        self.timestep_joined = 0  # Will be set by Platform

    def get_competing_score(self):
        '''Computes the sum of all competing attributes of the CC.'''
        kind_attributes_exp = self.config['kind_attributes_expanded']
        quality = 0
        for i, k in enumerate(kind_attributes_exp):
            if 'c' in k:
                quality += self.attributes.values[i]
        return quality

    def maybe_create_post(self, timestep):
        '''Random chance to create a post this iteration'''
        if not self.active:
            return False

        prob_post = self.config.get('prob_post', 0.25)
        if np.random.random() < prob_post:
            self.posts_recent += 1
            new_post_count = self.posts_recent
            self.posts_history.append((timestep, new_post_count))
            return True
        return False

    def update_activity_score(self):
        '''Calculate current activity score based on posts and engagement'''
        if not self.active:
            return 0.0

        alpha = self.config.get('alpha_activity', 1.0)
        beta = self.config.get('beta_activity', 2.0)

        # Engagement rate calculation (avoid division by zero)
        denom = max(1, self.total_followers_count)
        # k = 1.3 if not self.protected else 1
        engagement_rate = self.followers_in_current_step / denom

        # Calculate activity score
        activity_score = (alpha * self.posts_recent) + (beta * engagement_rate)
        return activity_score

    def evaluate_dropout(self):
        '''Determine if creator should become inactive'''
        # if self.active_time < 5:
        #     return False

        if not self.active:
            return False

        # Ensure there is a previous quality value to compare.
        if len(self.quality_history) < 2:
            return False

        activity_score = self.update_activity_score()
        threshold = self.config.get('dropout_threshold', 0.3)
        previous_quality = self.quality_history[-2]
        current_quality = self.current_quality

        k = 0.5
        # Logistic dropout probability
        dropout_prob = 1.0 / (1.0 + np.exp(-k * (threshold - activity_score)))
        growth = (current_quality - previous_quality) / (previous_quality + 1e-5)

        # If quality declines, force dropout.
        if growth < 0:
            self.active = False
            return True

        adjust_factor = 1.0 / (1.0 + np.exp(growth))
        dropout_prob *= adjust_factor

        adjusted_random = np.random.random() * np.exp(growth)

        if adjusted_random < dropout_prob:
            self.active = False
            return True
        return False

    def update_quality(self):
        '''Update quality based on learning and fatigue'''
        if not self.active:
            return

        # Get parameters
        lr = self.config.get('learning_rate', 0.01)
        lam = self.config.get('fatigue_lambda', 0.0005)

        # Update active time
        self.active_time += 1

        # Calculate learning and fatigue effects
        learning_factor = 1.0 + (lr * self.active_time)
        fatigue_factor = np.exp(-lam * self.active_time)
        new_quality = self.base_quality * learning_factor * fatigue_factor
        # Update current quality
        # self.current_quality = self.base_quality * learning_factor * fatigue_factor
        if not self.protected:
            new_quality = max(new_quality, self.current_quality)

        self.current_quality = new_quality
        self.quality_history.append(new_quality)
        self.quality_history.append(self.current_quality)

    def reset_counters_for_next_step(self):
        '''Reset per-iteration counters'''
        self.posts_recent = 0
        self.followers_in_current_step = 0

    def weight_followers_RS(self):
        '''Bias weight for recommender system'''
        if not self.active:
            return 0

        if self.protected:
            return 1 - self.config.get('level_bias_RS', 0)
        return 1


class Network:
    '''Class capturing a follower network between from users to items.
    In this version of the code we assumme that each item is a content creator/channel.
    '''

    def __init__(self, config, G=None, favorite=None):
        self.config = config

        num_users = config['num_users']
        num_items = config['num_items']

        self.G = G
        if self.G is None:
            self.G = np.zeros((num_users, num_items), dtype=bool)

        # self.favorite = favorite

        self.num_followers = np.count_nonzero(self.G, axis=0)
        self.num_followees = np.count_nonzero(self.G, axis=1)

    def follow(self, u, c, num_timestep, when_users_found_best):
        '''User u follows content creator c; and updates the Network

        input: u - user
               c - CC
               num_timestep - the iteration number of the platform (int)
               when_users_found_best - a list of length the number of users who keeps the timesteps when each of the user found their best CC (or -1 if they didn't yet)
        '''

        if not self.G[u.id][c.id]:
            if u.decide_follow(c):
                self.G[u.id][c.id] = True
                self.num_followers[c.id] += 1
                self.num_followees[u.id] += 1

                # if c is one of the best CCs for u, then u found their best CC this round
                if u.ranking_CCs[c.id] == 0:
                    when_users_found_best[u.id] = num_timestep

    def is_following(self, u, i):
        return self.G[u][i]


class RS:
    '''Class for the Recommender System (i.e., descoverability  procedure).
    '''

    def __init__(self, config, content_creators):
        self.config = config
        self.biased_weights = np.array(
            [c.weight_followers_RS() for c in content_creators])

    def recommend_random(self, content_creators, biased=False):
        """
        Chooses content creators uniformly at random.
        If biased is True, uses the biased weights for probability.
        """
        num_users = self.config['num_users']
        num_candidates = len(content_creators)

        if biased:
            # Get the indices of the active content creators
            active_indices = [cc.id for cc in content_creators]
            # Filter the biased weights for active candidates
            active_weights = self.biased_weights[active_indices]
            prob_choice = active_weights / np.sum(active_weights)
            if debug:
                print('Prob choice RS (biased):', prob_choice)
            # Sample with replacement to account for possibly fewer candidates than users
            return np.random.choice(content_creators, num_users, replace=True, p=prob_choice)

        # Unbiased: sample uniformly with replacement
        return np.random.choice(content_creators, num_users, replace=True)

    def recommend_PA(self, content_creators, num_followers, biased=False):
        num_users = self.config['num_users']
        # Build a list of active content creators and extract their IDs (indices)
        active_indices = [cc.id for cc in content_creators]
        # Filter the follower counts for these active creators
        active_followers = num_followers[active_indices]
        # Number of active candidates
        num_active = len(content_creators)

        # Calculate probability vector based only on active creators
        prob_choice = active_followers + np.ones(num_active)
        if biased:
            active_weights = self.biased_weights[active_indices]
            prob_choice *= active_weights
        prob_choice = prob_choice / np.sum(prob_choice)

        if debug:
            print('Probability vector for active candidates:', prob_choice)

        # Ensure sampling is done with replacement in case there are fewer candidates than users
        return np.random.choice(content_creators, num_users, replace=True, p=prob_choice)

    def recommendable_ExtremePA(self, content_creators, num_followers, biased=False):
        '''
        Input:
          content_creators - a list of content creator objects
          num_followers - a numpy array with follower counts (or probabilities) for each CC,
                          where the index corresponds to the creator's id.
          biased - True if the RS uses biased weights
        Output:
          A list of CCs that have the maximum weighted follower count.
        '''
        # Compute weighted follower counts (or simply use num_followers if un-biased)
        weighted_num_followers = self.biased_weights * num_followers if biased else num_followers

        # Get the maximum value from the full weighted array
        max_num_followers = max(weighted_num_followers)

        # Collect all content creators in the candidate list that match the maximum value.
        most_popular_CCs = [c for c in content_creators if weighted_num_followers[c.id] == max_num_followers]

        # Fallback: if no candidate in the active list reaches the max (or the list is empty),
        # pick the candidate with the highest weighted follower count.
        if not most_popular_CCs and content_creators:
            fallback_candidate = max(content_creators, key=lambda c: weighted_num_followers[c.id])
            most_popular_CCs = [fallback_candidate]

        return most_popular_CCs

    def recommend_ExtremePA(self, content_creators, num_followers, biased=False):
        ''' input: content_creators - a list of content creators
                   num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a CC chosen based on Extreme PA'''

        num_users = self.config['num_users']

        most_popular_CCs = self.recommendable_ExtremePA(
            content_creators, num_followers, biased)

        if biased:
            prob_choice = np.array([self.biased_weights[c.id]
                                    for c in most_popular_CCs])
            prob_choice = prob_choice / sum(prob_choice)
            if debug:
                print('Prob choice RS:', prob_choice)
            return np.random.choice(most_popular_CCs, num_users, p=prob_choice)
        return np.random.choice(most_popular_CCs, num_users)

    def recommend_AntiPA(self, content_creators, num_followers):
        ''' input: content_creators - a list of content creators
                   num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a CC chosen based on Anti-PA (nodes proportional to exp(-deg) )'''

        num_users = self.config['num_users']
        prob_choice = np.exp(-num_followers) / sum(np.exp(-num_followers))

        return np.random.choice(content_creators, num_users, p=prob_choice)

    def recommend(self, content_creators, num_followers):
        '''A rapper that choses the appropriate RS.

        input: content_creators - a list of content creators
               num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a list of reccommendations (one per user)'''

        if self.config['rs_model'] == 'UR':
            return self.recommend_random(content_creators)
        elif self.config['rs_model'] == 'PA':
            return self.recommend_PA(content_creators, num_followers)
        elif self.config['rs_model'] == 'ExtremePA':
            return self.recommend_ExtremePA(content_creators, num_followers)
        elif self.config['rs_model'] == 'biased_UR':
            return self.recommend_random(content_creators, biased=True)
        elif self.config['rs_model'] == 'biased_PA':
            return self.recommend_PA(content_creators, num_followers, biased=True)
        elif self.config['rs_model'] == 'biased_ExtremePA':
            return self.recommend_ExtremePA(content_creators, num_followers, biased=True)
        elif self.config['rs_model'] == 'AntiPA':
            return self.recommend_AntiPA(content_creators, num_followers)
        elif self.config['rs_model'] == 'PA-AntiPA':
            if np.random.random() < 0.5:
                return self.recommend_PA(content_creators, num_followers)
            return self.recommend_AntiPA(content_creators, num_followers)


# class Platform:
#     def __init__(self, config):
#         self.config = config
#
#         # the platform keeps track of the number of timesteps it has been iterated
#         self.timestep = 0
#
#         # make an expanded version of the kind of attributes
#         self.config['kind_attributes_expanded'] = []
#         for i, k in enumerate(config['kind_attributes']):
#             self.config['kind_attributes_expanded'] += [
#                 k] * config['num_attributes'][i]
#
#         if config['type_attributes'] == 'multidimensional':
#             self.config['covariance'] = self.construct_covariance()
#
#         self.network = Network(config)
#         self.CCs = []
#         self.generate_CCs()
#         self.RS = RS(config, self.CCs)
#         self.users = [User(config, i, self.CCs)
#                       for i in range(config['num_users'])]
#
#         # keep track of the timesteps when users found their best CC
#         self.users_found_best = [-1 for u in self.users]
#         # keep track of the position of the recommended CC in the ranking of the user
#         self.users_rec_pos = []
#         # keep track of whether or not the recommende CC had the same maching attributes
#         self.rec_same_maching = []
#         # keep track of the number of users recommended each CC in each round
#         self.num_users_rec_CC = []
#
#         # the users who did not converged yet
#         self.id_searching_users = list(range(self.config['num_users']))
#
#         if debug:
#             print('The users on the platform have attributes and preferences:')
#             for u in self.users:
#                 print('   ', u.id, u.attributes.values, u.weights)
#
#             print('The CCs on the platform are:')
#             for c in self.CCs:
#                 print('   ', c.id, c.attributes.values)
#
#     def generate_CCs(self):
#         '''Generates CCs that have attributes according to the config file'''
#
#         # 1. find if there is a restriction on the % of users of type A (not --> random)
#         per_groupA = self.config['%_groupA']
#         if per_groupA == -1:
#             self.CCs = [CC(self.config, i)
#                         for i in range(self.config['num_items'])]
#         # 2. else we keep adding CCs of the correct type
#         else:
#             # define the remaining number of users of each type
#             num_typeA = int(self.config['num_items'] * self.config['%_groupA'])
#             num_typeB = self.config['num_items'] - num_typeA
#             # find the first matching attribute (needs to exist)
#             protected_index = self.config['kind_attributes_expanded'].index(
#                 'm')
#
#             self.CCs = []
#             while num_typeA + num_typeB > 0:
#                 c = CC(self.config,
#                        self.config['num_items'] - num_typeA - num_typeB)
#                 if c.attributes.values[protected_index] == -1 and num_typeA:
#                     num_typeA -= 1
#                     self.CCs.append(c)
#                 elif c.attributes.values[protected_index] == 1 and num_typeB:
#                     num_typeB -= 1
#                     self.CCs.append(c)
#
#     def construct_covariance(self):
#         '''Constructs the covariance matrix'''
#
#         num_attributes = self.config['num_attributes']
#         dict_cov = self.config['dict_cov']
#
#         no_types = len(num_attributes)
#         total_num_attributes = sum(num_attributes)
#         cov = np.ones((total_num_attributes, total_num_attributes))
#
#         def type(i):
#             '''Given a index, i, returns the type of the attribute on position i.
#             (can be done faster)'''
#             sum = 0
#             for t in range(no_types):
#                 sum += num_attributes[t]
#                 if i < sum:
#                     return t
#             return no_types
#
#         # create the covariance matrix
#         for i in range(total_num_attributes):
#             # keep the diagonal (variance) 1; so start from i+1
#             for j in range(i + 1, total_num_attributes):
#                 cov[i, j] = dict_cov[(type(i), type(j))]
#                 cov[j, i] = dict_cov[(type(i), type(j))]
#
#         return cov
#
#     def iterate(self):
#         '''Makes one iteration of the platform.
#         Used only to update the state of the platform'''
#
#         # 0) the platform starts the next iteration
#         self.timestep += 1
#
#         # 1) each user gets a recommendation
#         recs = self.RS.recommend(self.CCs, self.network.num_followers)
#         # record the position of the recommended CC
#         self.users_rec_pos.append(
#             [self.users[i].ranking_CCs[c.id] for i, c in enumerate(recs)])
#         # record whether the user and the recommended CC mached on type
#         self.rec_same_maching.append(
#             [int(self.users[i].maching_attr == c.maching_attr) for i, c in enumerate(recs)])
#         # record the number of users recommended each CC
#         self.num_users_rec_CC.append([0 for c in self.CCs])
#         for c in recs:
#             self.num_users_rec_CC[-1][c.id] += 1
#
#         # 2) each user decides whether or not to follow the recommended CC
#         for u in self.users:
#             self.network.follow(
#                 u, recs[u.id], self.timestep, self.users_found_best)
#
#         if debug:
#             print('Recommendations: ', [r.id for r in recs])
#             print('New network:', self.network.G)
#             print('Number of followers:', self.network.num_followers)
#             print('Number of followees:', self.network.num_followees)
#
#     def get_borda_scores(self, rule='original'):
#         '''This reflects the global preferences of the consumers (users) on the creators (items).
#         1) each consumer ranks all creators (known & unknown);
#         2) they give points to each depending on the rule:
#              - original --> 1st position gets num_creators points, 2nd gets num_creators-1, ..., 1
#         ---
#         Returns: array with the borda score for each item
#            borda[i] = the score of item i (larger = better)
#         '''
#
#         num_items = self.config['num_items']
#         num_users = self.config['num_users']
#
#         # get the scores (each raw for a user)
#         scores = np.array([[u.score(c) for c in self.CCs] for u in self.users])
#
#         # get the preferences by sorting the scores
#         prefs = scores.argsort()  # each row has the respective user's items in increasing order
#
#         borda = np.zeros(num_items)
#         if rule == 'original':
#             for u in range(num_users):
#                 for s in range(num_items):
#                     borda[prefs[u][s]] += (s + 1)
#         elif rule == 'power':
#             for u in range(num_users):
#                 for s in range(num_items):
#                     borda[prefs[u][s]] += 1 / (num_items - s)
#
#         return borda
#
#     def get_competing_scores(self):
#         ''' Computes the quality of each CC.
#         -----
#         output: array with the competing socre (i.e., quality with equal weights for dim)
#         '''
#
#         quality = []
#         for c in self.CCs:
#             quality.append(c.get_competing_score())
#
#         return quality
#
#     def update_searching_users(self):
#         '''Updates the list of users who are still searching for the best CC.
#         i.e. those who did not find the best CC out of the ones that could be recommended
#         '''
#
#         if self.config['rs_model'] in ['ExtremePA', 'biased_ExtremePA']:
#             # under non-exploratory RSs, the searching users are only the ones who can still find somebody better
#
#             # 1) get the CCs with a maximum number of followers
#             most_popular_CCs = self.RS.recommendable_ExtremePA(
#                 self.CCs, self.network.num_followers, biased='biased' in self.config['rs_model'])
#
#             # 2) find if the user with id i converged
#             def converged(i):
#                 for c in most_popular_CCs:
#                     u = self.users[i]
#                     if u.score(c) > u.score(u.best_followed_CC):
#                         return True
#                 return False
#
#             # 3) filter users based on whether they can still find a better CC
#             self.id_searching_users = list(
#                 filter(converged, self.id_searching_users))
#         else:
#             # under exploratory RSs, the searching users are only the ones who did not find the best
#             self.id_searching_users = list(
#                 filter(lambda i: self.users[i].ranking_CCs[self.users[i].best_followed_CC.id] != 0, self.id_searching_users))
#
#     def check_convergence(self):
#         # the platform converged if there are no more searching users (users who can find better CCs)
#         return len(self.id_searching_users) == 0

import numpy as np


class Platform:
    """
    A class representing the simulation environment:
      - Content Creators (CCs) who can create posts, gain followers,
        update quality, and potentially drop out.
      - Users who follow recommended CCs.
      - A Recommender System (RS) that selects which CC each user sees each iteration.

    Steps in each iteration:
      1. Content creation phase: active CCs possibly make a new post.
      2. Recommendation phase: RS recommends an active CC to each user.
      3. User follow phase: each user decides whether to follow that recommended CC.
      4. CC update phase: each CC checks for dropout, updates its quality,
         and we log the new quality if still active.
      5. Reset counters for the next iteration (posts, new followers, etc.).
      6. Check for convergence (no active CCs or no more searching users).
    """

    def __init__(self, config):
        """
        Initializes the Platform with the given config.

        :param config: Dictionary of parameters controlling the simulation, e.g.:
          - num_users, num_items
          - type_attributes, num_attributes, kind_attributes
          - learning_rate, fatigue_lambda, dropout_threshold (for dynamic quality)
          - alpha_activity, beta_activity, prob_post (for activity)
          - etc.
        """
        self.config = config
        self.timestep = 0

        # ---------------------------------------------------------------------
        # 1) Expand attribute definitions for items & users if needed
        # ---------------------------------------------------------------------
        # Example logic for dimension expansion:
        self.config['kind_attributes_expanded'] = []
        for i, k in enumerate(config['kind_attributes']):
            self.config['kind_attributes_expanded'] += [k] * config['num_attributes'][i]

        # If using multivariate normal for attributes, build covariance
        if config['type_attributes'] == 'multidimensional':
            self.config['covariance'] = self.construct_covariance()

        # ---------------------------------------------------------------------
        # 2) Create the Network, CCs, Recommender, and Users
        # ---------------------------------------------------------------------
        self.network = Network(config)  # assumption: you have a Network class
        self.CCs = []
        self.generate_CCs()  # fill self.CCs with CC objects

        self.RS = RS(config, self.CCs)  # assumption: you have an RS class

        # Create users (each with an ID and references to CCs)
        self.users = [User(config, i, self.CCs) for i in range(config['num_users'])]

        # ---------------------------------------------------------------------
        # 3) Tracking Data
        # ---------------------------------------------------------------------
        # a) user-found-best tracking
        self.users_found_best = [-1 for _ in self.users]
        # b) recommendation logs
        self.users_rec_pos = []  # stores positions of recommended CC in user ranking
        self.rec_same_maching = []  # stores whether user & CC match on some attribute
        self.num_users_rec_CC = []  # number of times each CC is recommended in a timestep

        # c) dynamic quality logs
        self.active_CCs = set(range(len(self.CCs)))  # track IDs of creators still active
        self.dropout_history = []  # list of lists, each sub-list has IDs of newly dropped CCs at iteration
        self.quality_snapshots = []  # list of dicts, each dict: {cc_id: current_quality} for active CCs

        # Some simulations track which users are still searching
        self.id_searching_users = list(range(config['num_users']))

        # Optionally store initial join time for each CC
        for cc in self.CCs:
            cc.timestep_joined = 0

    # -------------------------------------------------------------------------
    # Placeholder: Covariance construction if needed
    # -------------------------------------------------------------------------
    def construct_covariance(self):
        """
        Builds the covariance matrix for the attributes if type_attributes == 'multidimensional'.
        This is typically copied from your existing code, e.g., reading self.config['dict_cov']
        and constructing an NxN matrix.
        """
        # Example placeholder
        if self.config['type_attributes'] != 'multidimensional':
            # Not needed; return None or identity
            return None

            # 1) Number of total attributes
        num_list = self.config['num_attributes']  # e.g. [1,1]
        total_num_attributes = sum(num_list)  # e.g. 2

        # 2) Load or parse dict_cov
        dict_cov = self.config['dict_cov']  # Could be a real dict or a string
        if isinstance(dict_cov, str):
            # If stored as a string, parse it
            dict_cov = ast.literal_eval(dict_cov)  # or json.loads if it's valid JSON

        # 3) Initialize NxN matrix
        cov = np.zeros((total_num_attributes, total_num_attributes), dtype=float)

        # 4) Fill it in from dict_cov
        for i in range(total_num_attributes):
            for j in range(total_num_attributes):
                if (i, j) not in dict_cov:
                    raise ValueError(f"Missing covariance entry (i={i}, j={j}). "
                                     f"Found keys: {list(dict_cov.keys())}")
                cov[i, j] = dict_cov[(i, j)]

        # 5) Optionally check if cov is positive semi-definite
        #    If it's degenerate or ill-defined, np.random.multivariate_normal can fail.
        #    For simplicity, we skip the check here.

        return cov

    # -------------------------------------------------------------------------
    # Placeholder: Generating CCs
    # -------------------------------------------------------------------------
    # def generate_CCs(self):
    #     """
    #     Populates self.CCs with content creator objects.
    #     Typically you'd do something like:
    #
    #         for i in range(self.config['num_items']):
    #             cc = CC(self.config, i)
    #             self.CCs.append(cc)
    #     """
    #     # Example placeholder
    #     pass

    def generate_CCs(self):
        num_items = self.config['num_items']
        frac_groupA = self.config.get('%_groupA', 0.5)  # fallback if not in config

        # Calculate how many CCs belong to group A
        groupA_count = int(num_items * frac_groupA)
        groupB_count = num_items - groupA_count

        # We'll create groupA_count creators with matching_attr = -1,
        # and groupB_count with matching_attr = +1, for instance.
        # This is just an example. Adapt to your 'Attributes' constructor logic.

        # groupA
        for i in range(groupA_count):
            cc = CC(self.config, id=i)
            # Force them to be 'protected' if your code identifies that by -1
            # e.g., cc.attributes.values[matching_index] = -1
            self.CCs.append(cc)

        # groupB
        for i in range(groupB_count):
            idx = groupA_count + i
            cc = CC(self.config, id=idx)
            # Force them to be unprotected if your code identifies that by +1
            self.CCs.append(cc)

    # -------------------------------------------------------------------------
    # Placeholder: Updating Searching Users
    # -------------------------------------------------------------------------
    def update_searching_users(self):
        """
        Removes users from 'id_searching_users' if they've already found their best possible CC.
        This is a naive approach; adapt it for your advanced logic if needed.
        """
        new_searchers = []
        for user_id in self.id_searching_users:
            user = self.users[user_id]
            # If the user has a best_followed_CC and it's top in ranking_CCs
            # 'ranking_CCs[c.id] == 0' typically means "rank #1"
            if user.best_followed_CC is not None:
                if user.ranking_CCs[user.best_followed_CC.id] == 0:
                    # They found their top; they can stop searching
                    continue
            # Otherwise, keep them searching
            new_searchers.append(user_id)

        self.id_searching_users = new_searchers

    # -------------------------------------------------------------------------
    # 4) Iteration / Main Simulation Loop
    # -------------------------------------------------------------------------
    def iterate(self):
        """
        Runs one iteration of the simulation:
          1. CCs maybe create posts
          2. RS recommends an active CC to each user
          3. Users follow the recommended CC
          4. CCs evaluate dropout, update quality; we log new qualities
          5. CC counters reset for next iteration
          6. We update 'id_searching_users' and check convergence

        :return: bool, True if we want to keep iterating, False if done
        """
        self.timestep += 1

        # ------------------- 1) Content Creation Phase -----------------------
        for cc in self.CCs:
            if cc.active:
                cc.maybe_create_post(self.timestep)

        # ------------------- 2) Filtering Active Content Creators ----------------
        # Gather all active content creators.
        active_CCs_list = [cc for cc in self.CCs if cc.active]
        if not active_CCs_list:
            # No active CC left, so stop iterating.
            return False

        # Compute activity scores for all active CCs.
        # (Assumes that update_activity_score() returns the current activity score based on posts and engagement.)
        activity_scores = np.array([cc.update_activity_score() for cc in active_CCs_list])
        # Retrieve protected status for each active CC.
        protected_flags = np.array([cc.protected for cc in active_CCs_list])

        # Filter for protected group:
        if np.any(protected_flags):
            protected_scores = activity_scores[protected_flags]
            # Determine the threshold at the 75th percentile (i.e., top 25% of activity)
            protected_threshold = np.percentile(protected_scores, 75)
            filtered_protected = [cc for cc, score in zip(active_CCs_list, activity_scores)
                                  if cc.protected and score >= protected_threshold]
        else:
            filtered_protected = []

        # Filter for unprotected group:
        if np.any(~protected_flags):
            unprotected_scores = activity_scores[~protected_flags]
            unprotected_threshold = np.percentile(unprotected_scores, 75)
            filtered_unprotected = [cc for cc, score in zip(active_CCs_list, activity_scores)
                                    if not cc.protected and score >= unprotected_threshold]
        else:
            filtered_unprotected = []

        # Combine filtered groups.
        filtered_active_CCs = filtered_protected + filtered_unprotected
        # Fallback: if filtering yields an empty list (e.g., not enough active CCs in either group), use all active CCs.
        if not filtered_active_CCs:
            filtered_active_CCs = active_CCs_list

        # Let the RS pick one CC for each user
        recs = self.RS.recommend(filtered_active_CCs, self.network.num_followers)

        # Record which CC was recommended for each user, for analysis
        # (1) positions in the user’s ranking
        self.users_rec_pos.append([
            self.users[i].ranking_CCs[c.id]
            for i, c in enumerate(recs)
        ])
        # (2) whether user & CC match in protected attribute
        self.rec_same_maching.append([
            int(self.users[i].maching_attr == c.maching_attr)
            for i, c in enumerate(recs)
        ])
        # (3) how many times each CC was recommended
        rec_counts = [0] * len(self.CCs)
        for rec in recs:
            rec_counts[rec.id] += 1
        self.num_users_rec_CC.append(rec_counts)

        # ------------------- 3) User Follow Phase ----------------------------
        # Each user decides to follow the recommended CC (if it improves utility)
        for u in self.users:
            self.network.follow(u, recs[u.id], self.timestep, self.users_found_best)

        # ------------------- 4) Creator Updates Phase ------------------------
        newly_inactive = []
        quality_snapshot = {}

        for cc in self.CCs:
            if cc.active:
                # Optionally capture quality before dropout (for final record)
                pre_drop_quality = cc.current_quality

                did_dropout = cc.evaluate_dropout()
                if did_dropout:
                    newly_inactive.append(cc.id)
                    if cc.id in self.active_CCs:
                        self.active_CCs.remove(cc.id)

                    # If you want the last known quality for dropped CC:
                    quality_snapshot[cc.id] = pre_drop_quality

                else:
                    # Remains active -> update quality
                    cc.update_quality()
                    quality_snapshot[cc.id] = cc.current_quality

        # Log which CCs dropped out in this iteration
        self.dropout_history.append(newly_inactive)
        # Log quality for active (or just-dropped) CCs
        self.quality_snapshots.append(quality_snapshot)

        # # ------------------- 5) Reset Counters -------------------------------
        # for cc in self.CCs:
        #     cc.reset_counters_for_next_step()

        # ------------------- 6) Convergence Check ----------------------------
        # Possibly refine 'id_searching_users' if your logic requires
        self.update_searching_users()

        # Return whether we want to continue iterating
        return not self.check_convergence()

    # -------------------------------------------------------------------------
    # 5) Convergence Logic
    # -------------------------------------------------------------------------
    def check_convergence(self):
        """
        The simulation ends if:
          - No CCs remain active, OR
          - No users are still searching (depending on your design).
        """
        no_active_left = (len(self.active_CCs) == 0)
        no_searching_users = (len(self.id_searching_users) == 0)
        return no_active_left or no_searching_users

    def get_borda_scores(self, rule='original'):
        """
        Reflects the global preferences of all users over the creators.
        The procedure typically:
          1) Each user ranks all creators from best to worst.
          2) The top-ranked creator gets the highest points, next gets fewer, etc.
             - "original" Borda: if there are N creators, top rank gets N points, 2nd gets N-1, ...
             - "power" Borda: you might use 1/(N - rank).

        :param rule: "original" or "power" or any other scoring scheme you define.
        :return: a list (or np.array) where each index i corresponds to the Borda score of creator i.
        """
        num_items = self.config['num_items']
        num_users = self.config['num_users']

        # scores[u][c] = user u's personal "score" for creator c
        scores = []
        for u in self.users:
            row = []
            for c in self.CCs:
                row.append(u.score(c))
            scores.append(row)

        scores = np.array(scores)  # shape: (num_users, num_items)

        # For each user, we get a preference order:
        # sorts columns in ascending order, so the best is at the end
        prefs = scores.argsort(axis=1)  # shape: (num_users, num_items)

        borda = np.zeros(num_items, dtype=float)
        if rule == 'original':
            # Each user awards N points to their #1, N-1 to their #2, etc.
            # So if sorted in ascending order, the last is rank #1.
            for u in range(num_users):
                for rank_pos in range(num_items):
                    item_id = prefs[u][rank_pos]
                    # The first in 'prefs[u]' is the user's worst, the last is the best
                    # rank_pos=0 => worst => gets 1 point
                    # rank_pos=(num_items-1) => best => gets N points
                    # So awarding points can be: rank_pos+1
                    # or do (num_items - rank_pos)
                    # It's your choice how you define Borda "original."
                    borda[item_id] += (rank_pos + 1)  # or use (num_items - rank_pos)
        elif rule == 'power':
            # e.g. we could do a harmonic scheme
            for u in range(num_users):
                for rank_pos in range(num_items):
                    item_id = prefs[u][rank_pos]
                    # The best (rank_pos = num_items-1) gets e.g. 1/1, second gets 1/2, etc.
                    # This is a matter of definition. Another approach:
                    # rank_pos=0 => 1/(num_items), rank_pos=1 => 1/(num_items-1), ...
                    # We'll just do 1/(num_items - rank_pos)
                    borda[item_id] += 1.0 / (num_items - rank_pos)
        else:
            # Default or fallback
            pass

        return borda

    def get_competing_score(self):
        # sum 'c' attributes
        total = 0
        kind_attributes_exp = self.config['kind_attributes_expanded']
        for i, kind in enumerate(kind_attributes_exp):
            if 'c' in kind:
                total += self.attributes.values[i]  # or something similar
        return total

    def get_competing_scores(self):
        # return list of get_competing_score() for each CC
        return [cc.get_competing_score() for cc in self.CCs]
