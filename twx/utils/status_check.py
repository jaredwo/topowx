'''
Created on Jul 5, 2011

@author: jared.oyler
'''
import time
import sys

class status_check(object):
    '''
    classdocs
    '''


    def __init__(self,total_cnt,check_cnt):
        '''
        Constructor
        '''
        self.total_cnt = total_cnt
        self.check_cnt = check_cnt
        self.num = 0 
        self.num_last_check = 0
        self.status_time = time.time()
        self.start_time = self.status_time
    
    def increment(self,n=1):
        self.num+=n
        if self.num - self.num_last_check >= self.check_cnt:
            currentTime = time.time()
            
            if self.total_cnt != -1:
                print "Total items processed is %d.  Last %d items took %f minutes. %d items to go."%(self.num,self.num - self.num_last_check,(currentTime - self.status_time)/60.0,self.total_cnt-self.num)
                print "Current total process time: %f minutes"%((currentTime - self.start_time)/60.0)
                print "Estimate Time Remaining: %f"%(((self.total_cnt-self.num)/float(self.num))*((currentTime - self.start_time)/60.0))
            
            else:
                print "Total items processed is %d.  Last %d items took %f minutes"%(self.num,self.num - self.num_last_check,(currentTime - self.status_time)/60.0)
                print "Current total process time: %f minutes"%((currentTime - self.start_time)/60.0)
            sys.stdout.flush()
            self.status_time = time.time()
            self.num_last_check = self.num
            
class timer(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self,msg_prfx):
        print "".join([msg_prfx," %f secs."%(time.time()-self.start_time)])
