'''
Created on May 21, 2010

@author: jared.oyler
'''

import processing
import time


class multiprocess_config():
    def __init__(self):
        self.status_check_num = None
        self.numProcs = None
        self.workerName = None
        self.outputHandler = None
        self.inQueueLimit = None


class multiprocess():
    '''
    classdocs
    '''
    def __init__(self, configMultiProcess,configWorker):
        '''
        Constructor
        '''
        self.config = configMultiProcess
        self.StatusCheck = configMultiProcess.status_check_num
        self.outputHandler = configMultiProcess.outputHandler
        if self.config.numProcs > 1:
            if self.config.inQueueLimit is not None:
                self.inq = processing.Queue(self.config.inQueueLimit)
            else:
                self.inq = processing.Queue()   
            self.outq = processing.Queue()
            #spawn worker processes
            self.workerProcs = []
            for i in range(self.config.numProcs):
                
                self.workerProcs.append(self.build_worker(configMultiProcess.workerName,self.inq,self.outq,configWorker))
                
            for p in self.workerProcs:
                p.start()
        else:
            self.singleWorker = self.build_worker(configMultiProcess.workerName,None,None,configWorker)
        self.numPredictPts = 0
        self.numPredictPtsLastStatus = 0
        self.queue_statusTime = None
        self.queue_startTime = None
    
    def build_worker(self,worker_name,inq,outq,config_worker):
        pass
    
    def process(self,predictPt):
        if self.numPredictPts == 0:
            self.queue_statusTime = time.time()
            self.queue_startTime = self.queue_statusTime
        
        if self.config.numProcs > 1:
            self.inq.put(predictPt)
        else:
            self.outputHandler.handleOutput(self.singleWorker.do_work(predictPt))
        self.numPredictPts+=1
        
        if self.StatusCheck != -1:
            if self.numPredictPts - self.numPredictPtsLastStatus == self.StatusCheck:
                currentTime = time.time()
                print "Total pts put on queue %d.  Last %d pts took %f minutes."%(self.numPredictPts,self.StatusCheck,(currentTime - self.queue_statusTime)/60.0)
                print "Current total process time for queue loading: %f minutes"%((currentTime - self.queue_startTime)/60.0)
                self.queue_statusTime = time.time()
                self.numPredictPtsLastStatus = self.numPredictPts
              
        
    def handleOutputs(self):
        
        if self.config.numProcs > 1:
            numProcessed = 0 
            numProcessedLastStatus = 0
            statusTime = time.time()
            startTime = statusTime
            while numProcessed < self.numPredictPts: 
                self.outputHandler.handleOutput(self.outq.get())
                numProcessed+=1
                if (self.StatusCheck != -1) and (numProcessed - numProcessedLastStatus == self.StatusCheck):
                    currentTime = time.time()
                    print "Total processed pts is %d.  Last %d pts took %f minutes. %d points to go."%(numProcessed,self.StatusCheck,(currentTime - statusTime)/60.0,self.numPredictPts-numProcessed)
                    print "Current total process time: %f minutes"%((currentTime - startTime)/60.0)
                    print "Estimate Time Remaining: %f"%(((self.numPredictPts-numProcessed)/self.StatusCheck)*((currentTime - statusTime)/60.0))
                    statusTime = time.time()
                    numProcessedLastStatus = numProcessed
            
    def terminate(self):
        if self.config.numProcs > 1:
            self.inq.close()
            self.outq.close()
            for p in self.workerProcs:
                p.terminate()
                del p
            del self.inq
            del self.outq

class worker(processing.Process):
    '''A base class for multithreaded operations'''
    def __init__(self, inq, outq):
        if inq is not None and outq is not None:
            self.inq = inq #the set of input data
            self.outq = outq #the location for output results
            processing.Process.__init__(self) #call the processing.Process constructor
    def run(self):
        while True:
            #get an item from the work queue
            item = self.inq.get()
            #None items signal the we have reached the end of the work to be done
            if item is None:
                break
        
            out = self.do_work(item)
            self.outq.put(out)
        self.outq.put(None) #this is an easy way to know when all the processes are done
        self.inq = None
        self.outq = None
        return()
    def do_work(self, input):
        pass 


