from pickle_ops import pickle_load

if __name__ == "__main__":
    final = pickle_load('bayesian_final.pkl')
    intermediate = pickle_load('bayesian_intermediate.pkl')

    print(final)
    print(len(intermediate))
    print(intermediate)