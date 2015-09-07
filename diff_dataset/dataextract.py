import numpy as np
def read_data(fn):
    """ read dataset and separate into input data
        and label data
    """
 
    # read dataset file
    with open(fn) as f:
        raw_data = np.loadtxt(f, delimiter= ',', dtype="float", 
            skiprows=1, usecols=None)
    # initilize list
    data = []; label = []
    #assign input data and label data
    for row in raw_data:
        data.append(row[:-1])
        label.append(int(row[-1]))
    # return input data and label data
    return np.array(data), np.array(label)

if __name__=='__main__':
    dataFile=raw_input('Enter Dataset File name:')
    dataset=read_data(dataFile)
    print 'No. of training set :',len(dataset[0])
    print 'No. of Input feature :',len(dataset[0][0])
    try:
        print 'No. of output/input set:',len(dataset[1][0])
    except:
        print 1
