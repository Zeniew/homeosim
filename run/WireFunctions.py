import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os

def whereGol(g,xGol): # returns the Golgi x,y position in layer
    y = g // xGol
    x = g % xGol
    return x,y
def distance(x1,y1,x2,y2,xGol,yGol):  # returns Euclidian distance between two Golgi cells in layer
    col_dist = min(abs(x2 - x1), xGol - abs(x2 - x1))
    row_dist = min(abs(y2 - y1), yGol - abs(y2 - y1))
    d = cp.sqrt((col_dist*col_dist)+(row_dist*row_dist))
    return d
def get_dist_arr(verbose, xGol=32,yGol=128):
    numGol = xGol * yGol
    filename = f"Gol_{xGol}by{yGol}_dist.npy"
    if os.path.exists(filename): # no need to read in data again
        print("Loading distances matrix...") if verbose else None
        dist = cp.load(filename)
    else:
        print("Building distances matrix...") if verbose else None
        dist = cp.zeros((numGol,numGol),dtype=float)   # distance between each Golgi pair
        for g1 in range(0,numGol):  # calculate distance between each Golgi pair
            x1,y1 = whereGol(g1,xGol)
            for g2 in range(0,numGol):
                x2, y2 = whereGol(g2,xGol)
                d = distance(x1,y1,x2,y2,xGol,yGol)
                dist[g1,g2]= d
        cp.save(filename, dist)
    return dist
def format_connect(Gconnect, numGol, conv):
    ### Formatting Connect array to match system arcitecture
    connect_arr = cp.zeros((numGol, conv), dtype=int)
    # transpose to have format be (Pre, Post)
    Gconnect = Gconnect.T
    for neuron in range(numGol):
        connect_idx = cp.where(Gconnect[neuron] == 1)[0]
        connect_arr[neuron] = connect_idx
    return connect_arr

def wire_up(conv, recip, span, verbose, showWhat=None, xGol=32, yGol=128):
    '''Convergence per Cell, Reciprical Connections per Cell, Radius of Span.
    ShowWhat... 1 = Span; 2 = Connections; 3 = Pre left'''
    dist = get_dist_arr(verbose)
    verify = False
    numGol = xGol * yGol
    # connectivity, 1 = connected
    Gconnect = cp.zeros((numGol,numGol),dtype=int)
    # counts number of connections that cell has made
    SynNum = cp.zeros(numGol,dtype = int)

    # Make all reciprocal connections first
    GolRandom = cp.arange(numGol)
    for cNum in range(recip): 
        cp.random.shuffle(GolRandom)
        for i in range(numGol):
            post = GolRandom[i]
            # eligible connections within span
            couldBePre = cp.where((dist[post,:] < span) & (dist[post,:]>0))[0] 
            numcandidates = len(couldBePre)
            cp.random.shuffle(couldBePre)
            if SynNum[post] <= cNum: # in case it's already gotten a synapse this round as post target
                for attempt in range(numcandidates):
                    possiblePre = couldBePre[attempt]
                    # don't have to check reciprocal because only reciprocals exist right now
                    if (Gconnect[post,possiblePre] == 0) and (SynNum[possiblePre] < recip):
                        #change both because connections are reciprocal
                        SynNum[post]+=1
                        SynNum[possiblePre]+=1
                        Gconnect[post,possiblePre]=1
                        Gconnect[possiblePre,post]=1
                        break
    # connect closest unconnected neurons that dont have enough recip connections
    problems = cp.where(SynNum!=recip)[0]
    if len(problems) > 0:
        print(f"Fixing {len(problems)} recip connections to nearest neighbor...") if verbose else None
        for i in range(0,len(problems),2):
            #print("fix",problems[i],problems[i+1])
            SynNum[problems[i]] +=1
            SynNum[problems[i+1]] +=1
            Gconnect[problems[i],problems[i+1]]=1
            Gconnect[problems[i+1],problems[i]]=1
    ### Non-reciprical connections
    ### Choose a post synaptic cell and pick a cell to connect to it
    # counts number of connections that cell has made
    preSynNum = cp.array(SynNum)
    if recip < conv:
        ### Initial wiring attempt
        for cNum in range(recip,conv): 
            cp.random.shuffle(GolRandom)
            for i in range(numGol):
                post = GolRandom[i]
                # get eligible connections within span that aren't itself
                allPreCandidates = cp.where((dist[post,:] < span) & (dist[post,:]>0))[0]
                # need to shuffle so not secondary ranked by distance
                cp.random.shuffle(allPreCandidates)
                # rank order possible pre by how many pre connections they've made and try to connect lowest first
                numLeft = preSynNum[allPreCandidates]
                sortedindeces = cp.argsort(numLeft)
                rankedPres = allPreCandidates[sortedindeces]
                for attempt in range(len(rankedPres)):
                    possiblePre = rankedPres[attempt]
                    # check connection both ways to make sure it doesn't become reciprical
                    if (Gconnect[post,possiblePre] == 0) and (Gconnect[possiblePre,post] == 0):
                        # check pre still has synapses to make
                        if preSynNum[possiblePre] < conv:
                            SynNum[post]+=1
                            preSynNum[possiblePre] +=1
                            Gconnect[post,possiblePre]=1
                            break
        ### Fixing cells under convergence                
        numToFix = cp.sum(SynNum < conv)
        print(f"Fixing {numToFix} non-recip connections by increasing span until made...") if verbose else None
        growSpan = span # new var for growing span
        cp.random.shuffle(GolRandom)
        # fix unmade connections by increasing span until all are made
        while numToFix > 0:
            growSpan *= 1.1
            for i in range(numGol):
                post = GolRandom[i]
                # needed if statement bc loop iterates over all cells every time
                if SynNum[post] < conv:
                    x = cp.where((dist[post,:] < growSpan) & (dist[post,:]>0))[0]
                    cp.random.shuffle(x)
                    # rank ordering by current num of pre connections
                    numLeft = preSynNum[x]
                    sortedindeces = cp.argsort(numLeft)
                    rankedPres = x[sortedindeces]
                    for attempt in range(len(rankedPres)):
                        possiblePre = rankedPres[attempt]
                        if (Gconnect[post,possiblePre] == 0) and (Gconnect[possiblePre,post] == 0):
                            if preSynNum[possiblePre] < conv:
                                SynNum[post]+=1
                                preSynNum[possiblePre] +=1
                                Gconnect[post,possiblePre]=1
                                numToFix -= 1
                                break
    ### Bug check
    if any(cp.sum(Gconnect, axis=1)!=conv):
        unfilled = cp.where(cp.sum(Gconnect, axis=1)!=conv)[0]
        unfilled_picked = set(cp.asnumpy(unfilled).tolist())
        print(f"Cell:{unfilled_picked} has {cp.sum(Gconnect, axis=1)[unfilled]} num connections.") if verbose else None
    elif not any(SynNum!=conv) and not any(preSynNum!=conv):
        verify = True
    
    ### Plotting Connect Arrays built into function
    if showWhat:
        plotX = cp.zeros(numGol)
        plotY = cp.zeros(numGol)
        plt.figure(figsize=(16, 5))
        for i in range(0,numGol):
            plotX[i],plotY[i]=whereGol(i,xGol)
        plt.scatter(plotY,plotX,color = 'lightgray')
        if showWhat == 1:
            Gol = 2000
            Golx, Goly = whereGol(Gol,xGol)
            plt.scatter(Goly,Golx,color = "black")
            for i in range(0,len(couldBePre)):
                x,y =whereGol(couldBePre[i],xGol) 
            x = cp.where((dist[Gol,:] < span) & (dist[Gol,:]>0))[0] # eligible connections within span
            numcandidates = len(x[0])-1
            couldBePre = cp.array(x[0,0:numcandidates+1])
            for i in range(0,len(couldBePre)):
                x,y = whereGol(couldBePre[i],xGol)
                plt.scatter(y,x,color="red")
            plt.show()
        if showWhat == 2:
            showGol = []
            for i in range(0,13):
                showGol.append(200 + (i*304))
            for j in range(0,len(showGol)):
                Gol = showGol[j]
                Golx, Goly = whereGol(Gol,xGol)
                plt.scatter(Goly,Golx,color = "black")
                for i in range(0,numGol):
                    if Gconnect[i,Gol] == 1:
                        x,y = whereGol(i,xGol)
                        plt.scatter(y,x,color="red")
                for i in range(0,numGol):
                    if Gconnect[Gol,i] == 1:
                        x,y = whereGol(i,xGol)
                        if Gconnect[i,Gol]==1:  # reciprocal
                            plt.scatter(y,x,color="purple")
                        else:
                            plt.scatter(y,x,color="blue")
            plt.show()
        if showWhat == 3:
            for i in range(0,numGol):
                Golx, Goly = whereGol(i,xGol,yGol)
                if preSynNum[i] == conv-1:
                    plt.scatter(Goly,Golx,color = "black")
                elif preSynNum[i] < conv-2:
                    plt.scatter(Goly,Golx,color = "red")
            plt.show()

    return Gconnect, verify

def wire_up_verified(conv, recip, span, showWhat=None, xGol=32, yGol=128, verbose=True):
    verify = False
    while verify == False:
        connect_arr, verify = wire_up(conv, recip, span, verbose, showWhat=showWhat, xGol=xGol, yGol=yGol)
        if verbose:
            if verify == False:
                print("Checked Failed, running again...")
            if verify == True:
                print("Connect Check Passed.")
    connect_formatted = format_connect(connect_arr, xGol*yGol, conv)
    return connect_formatted