#-----------------------------------------------------------------------------
#PHY3042
#Assignment 2
#Student: 6463360
#Task 1
#---Programme Description-----------------------------------------------------
#This programme takes a truth table and trains a neural network to output
#weights that will give the correct values to the table. There is an
#accompanying document for further clarification.
#-----------------------------------------------------------------------------
#Import useful libraries.
import numpy as np
from datetime import datetime


#-Functions-------------------------------------------------------------------
def act(x):
    k = 20.0
    return 1.0 / (1.0 + np.exp(-k*x))

def error(w):   #3 hidden nodes, input passes to out too.
    out = [0, 0, 0, 0, 0, 0, 0]
    local_err = [0, 0, 0, 0, 0, 0, 0]
    for i in range(7):
        i1 = tt[i][0]; i2 = tt[i][1]; i3 = tt[i][2]
        h4 = act(w[0] + w[1]*i1 + w[2]*i2 + w[3]*i3)
        h5 = act(w[4] + w[5]*i1 + w[6]*i2 + w[7]*i3)
        h6 = act(w[8] + w[9]*i1 + w[10]*i2 + w[11]*i3)
        o7 = act(w[12] + w[13]*i1 + w[14]*i2 + w[15]*i3 + w[16]*h4 + \
                 w[17]*h5 + w[18]*h6)
        out[i] = o7
        local_err[i] = (out[i] - tt[i][3])**2
    error = 0.5 * sum(local_err)
    return error, out


#-Main------------------------------------------------------------------------
#Open event log file.
log = open("task_1_event_log.txt", 'w')
log.write("""Task 1 Event Log - Student 6463360
\n""")


#Define the variables.
tt = [ [0, 0, 0, 0],    #Truth table.
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0] ]

#Start by starting weights randomly.
beta = 0.01
N = 19    #Number of weights in the network.
w = 2.0 * np.random.random(N) - 1.0
init_w = w.copy()
init_err, out = error(w)

#Loop variables (counters, errors, etc).
old_err, new_err = 1, 1
i, j = 0, 1


#Starting message.
print("""
**Please read the accompanying document for further information.**
This neural network learns how to reproduce the following truth table:
I1 I2 I3 Out""")
for i in range(7):
    print(tt[i])
print("""
Learning...""")

#Run tests and change weights randomly while the case is not saticfied.
while old_err > 1e-6:  #Loop until error condition is met.
    
    #Vary beta with each loop to reduce chance of accepting worse weights.
    beta += 1
    
    #Write to the logfile.
    log.write("""{}
Attempt {}: Run {}\n""" .format(str(datetime.now()), j, i))
    
    #Calculate error for configuration of weights.
    old_err, out = error(w)
    
    #Make 1 change to a random weight.
    wnew = w.copy()
    wnew[np.random.randint(0, len(w) - 1)] = (2.0 * np.random.random(1) - 1.0)
    
    #Find new error.
    new_err, out = error(wnew)
    log.write("""Old weights = {}
New weights = {}
Old error = {}
New error = {}\n""" .format(w, wnew, old_err, new_err))
    
    #Compare errors.
    if new_err < old_err:
        w = wnew.copy()
        log.write("New error lower than old. Accepting new weights.\n")
    else:
        prob = np.exp((old_err - new_err) * beta)
        rand = np.random.random(1)
        if rand < prob:
            #Longer but accepting.
            w = wnew.copy()
            log.write("""New error larger than old - accept new weights anyway.
Probability = {} and random number = {}\n""" .format(prob, rand[0]))
        else:
            log.write("Not accepting new weights.\n")
    
    #Add space in log file.
    log.write("\n")
    
    #Incriment run counter and display to user.
    i += 1
    print("Number of runs =", i, end='\r')
print("Number of runs =", i)


#Close the Event Log file.
log.close()


#Print loads of useful stuff to the user (try and make it look neat).
print("""Network trained sucessfully.
""")

#Print old and new wights for comparison.
print("Weights: Initial    Final")
for i in range(N): print("w[%s] =" %i, '%.6f' %init_w[i], "", '%.6f' %w[i])
print("")

#Print final truth table.
#Formatted to look as nice as possible on the terminal.
#Terminal allignment of columns may vary due to being developed on Windows.
old_err, out = error(w)
print("""Initial error = {}
Final error = {}

Resulting truth table:
I1 : I2 : I3 : Out : Expected""" .format('%.9f' %init_err, '%.9f' %old_err))
for i in range(7):
    print(tt[i][0], " :", tt[i][1], " :", tt[i][2], " :", round(out[i]), \
          "  :", tt[i][3])

#Final message to user.
print("""
An event log file has been created called: task_1_event_log.txt
It contains the run by run information of the weights and errors.""")