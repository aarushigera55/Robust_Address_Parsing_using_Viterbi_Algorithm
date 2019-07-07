import math
from operator import itemgetter

def find_symbols(line):
    f=''
    for i in range(0,len(line)):
        if line[i] is '(' or line[i] is ')' or line[i] is '&' or line[i] is '/' or line[i] is '-' or line[i] is ',':
            f += ' ' + line[i] + ' '
        else:
            f += line[i]
    line = f.split()
    return line


def calc_a_and_b(State_File, Symbol_File):

    #  Retrieve data from StateFile
    M=0
    N=0
    with open(State_File) as s:
        ctr = 0  # counter for lines
        states = []
        for line in s:
            if ctr == 0:
                N = int(line.strip())
                state_by_state = [[0]*N for _ in range(N)]
            else:
                if ctr < N+1:
                    states.append(line.strip())
                else:
                    temp = [int(i) for i in (line.strip().split())]
                    state_by_state[temp[0]][temp[1]] = temp[2]
            ctr += 1
    s.close()

    Prob_A = []
    for i in range(N):
        temp = [(j + 1)/(sum(state_by_state[i]) + N - 1) for j in state_by_state[i]]
        Prob_A.append(temp)

    st_state = N-2
    end_state = N-1

    for i in range(N):
        Prob_A[i][st_state]=0
    Prob_A[end_state]=[0 for j in range(N)]

    with open(Symbol_File) as s:
        ctr = 0  # counter for lines
        symbols = []
        for line in s:
            if ctr == 0:
                M = int(line.strip())
                state_by_symbol = [[0] * M for i in range(N-2)]
                # print(state_by_symbol)
            else:
                if ctr < M + 1:
                    symbols.append(line.strip())
                else:
                    temp = [int(i) for i in (line.strip().split())]
                    state_by_symbol[temp[0]][temp[1]] = temp[2]
            ctr += 1
    s.close()
    Prob_B = []
    for i in range(N-2):
        S = (sum(state_by_symbol[i]) + M + 1)
        Prob_B.insert(i, [(j + 1) / S for j in state_by_symbol[i]])
        Prob_B[i].append(1 / S)

    return Prob_A, Prob_B, states, symbols, N, M

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File):  # do not change the heading of the function

    Prob_A, Prob_B, states, symbols, num_of_states, num_of_symbols = calc_a_and_b(State_File, Symbol_File)
    output = []
    with open(Query_File) as s:
        for line in s:
            line = find_symbols(line)
            symbol_index = []
            for i in line:
                if i in symbols:
                    symbol_index.append(symbols.index(i))
                else:
                    symbol_index.append(len(symbols))
            Viterbi_Probs = viterbi(Prob_A, Prob_B, symbol_index, num_of_states)
            log_probaility = math.log(Viterbi_Probs[-1][num_of_states - 1]["prob"])
            j = num_of_states - 1
            for i in range(len(Viterbi_Probs) - 1, -1, -1):
                prev = Viterbi_Probs[i][j]["prev"]
                Viterbi_Probs[i] = j
                j = prev
            Viterbi_Probs.append(log_probaility)
            Viterbi_Probs.insert(0, num_of_states - 2)
            output.append(Viterbi_Probs)
    s.close()
    return output

def viterbi(Prob_A, Prob_B, symbol_index, num_of_states):
    V = fill_prob(symbol_index,num_of_states,Prob_A,Prob_B)

    # final probability going to the end state by merging the max

    V.append({})
    maximum_trans = Prob_A[0][num_of_states - 1] * V[-2][0]["prob"]
    next_prev_state = 0
    for i in range(1, num_of_states - 2):
        max_prob_A_check = Prob_A[i][num_of_states-1] * V[-2][i]["prob"]
        if maximum_trans > max_prob_A_check:
            pass
        else:
            next_prev_state = i
            maximum_trans = max_prob_A_check
    V[-1][num_of_states-1] = {"prob": maximum_trans, "prev": next_prev_state}
    return V

def fill_prob(symbol_index,num_of_states,Prob_A,Prob_B):

    # Setup the first state to all states
    V = [{}]
    for st in range(0, num_of_states - 2):
        V[0][st] = {"prob": Prob_A[num_of_states - 2][st] * Prob_B[st][symbol_index[0]], "prev": None}

    # Put probabilities based on the other state and Vierbi probabilities
    for i in range(1, len(symbol_index)):
        V.append({})
        for j in range(0,num_of_states-2):
            maximum_trans = Prob_A[0][j] * V[i-1][0]["prob"]
            next_prev_state = 0
            for k in range(1,num_of_states-2):
                max_prob_A_check = Prob_A[k][j] * V[i-1][k]["prob"]
                if maximum_trans > max_prob_A_check:
                    pass
                else:
                    next_prev_state = k
                    maximum_trans = max_prob_A_check

            if symbol_index[i] != num_of_states:
                V[i][j] = {"prob": maximum_trans*Prob_B[j][symbol_index[i]], "prev": next_prev_state}
    return V

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k):  # do not change the heading of the function

    '''
        The only thing we change here compared to q1 is the calling on a different Viterbi method.
        This is because we will now only account for the top-k elements and not check all the previous states for max probability.
    '''

    Prob_A, Prob_B, states, symbols, num_of_states, num_of_symbols = calc_a_and_b(State_File, Symbol_File)
    output = []
    with open(Query_File) as s:
        for line in s:
            line = find_symbols(line)
            symbol_index = []
            for i in line:
                if i in symbols:
                    symbol_index.append(symbols.index(i))
                else:
                    symbol_index.append(len(symbols))

            Viterbi_Probs_Top_K = viterbi_top_k(Prob_A, Prob_B, symbol_index, num_of_states, k)
            for i in range(0,k):
                Viterbi_Probs_Top_K[i][-1] = math.log(Viterbi_Probs_Top_K[i][-1])
            output += Viterbi_Probs_Top_K
    s.close()
    return output

def viterbi_top_k(Prob_A, Prob_B, symbol_index, num_of_states, k):

    V = fill_prob_top_k(symbol_index, num_of_states, Prob_A, Prob_B, k)

    L = []
    for i in V:
        for j in V[i]:
            x = j["prev"][:]
            x.append(num_of_states-1)
            x.append(Prob_A[i][num_of_states-1] * j["prob"])
            L.append(x)
    L = sorted(L)
    x = len(L[0]) - 1
    L = sorted(L, key=itemgetter(x), reverse=True)
    L = L[:k]

    return L



def fill_prob_top_k(symbol_index, num_of_states, Prob_A, Prob_B, k):

    # Setup the first state to all states
    V = [{}]
    for st in range(0, num_of_states - 2):
        V[0][st] = [
            {"prob": Prob_A[num_of_states - 2][st] * Prob_B[st][symbol_index[0]], "prev": [num_of_states - 2, st]}]

    # Put probabilities based on the other state and Vierbi probabilities
    for i in range(1, len(symbol_index)):
        V.append({})
        for j in range(0, num_of_states - 2):
            L = []
            for p in V[i - 1]:
                for q in V[i - 1][p]:
                    x = q["prev"][:]
                    x.append(j)
                    L.append({"prob": Prob_B[j][symbol_index[i]] * Prob_A[p][j] * q["prob"], "prev": x})

            # As our list now has all the combinations of the previous top k * current transitions we will limit the new list by first k as it is already in ascending order of the path it is taking
            L = sorted(L, key=itemgetter("prob"), reverse=True)
            if len(L) > k: L = L[:k]
            V[i][j] = L
    return V[-1]

# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File):  # do not change the heading of the function
    pass  # Replace this line with your implementation...
