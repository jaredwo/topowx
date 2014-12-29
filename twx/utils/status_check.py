'''
Copyright 2014, Jared Oyler.

This file is part of TopoWx.

TopoWx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TopoWx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.
'''
import time
import sys

class StatusCheck(object):
    '''
    classdocs
    '''


    def __init__(self, total_cnt, check_cnt):
        '''
        Constructor
        '''
        self.total_cnt = total_cnt
        self.check_cnt = check_cnt
        self.num = 0 
        self.num_last_check = 0
        self.status_time = time.time()
        self.start_time = self.status_time
    
    def increment(self, n=1):
        self.num += n
        if self.num - self.num_last_check >= self.check_cnt:
            currentTime = time.time()
            
            if self.total_cnt != -1:
                print "Total items processed is %d.  Last %d items took %f minutes. %d items to go." % (self.num, self.num - self.num_last_check, (currentTime - self.status_time) / 60.0, self.total_cnt - self.num)
                print "Current total process time: %f minutes" % ((currentTime - self.start_time) / 60.0)
                print "Estimated Time Remaining: %f" % (((self.total_cnt - self.num) / float(self.num)) * ((currentTime - self.start_time) / 60.0))
            
            else:
                print "Total items processed is %d.  Last %d items took %f minutes" % (self.num, self.num - self.num_last_check, (currentTime - self.status_time) / 60.0)
                print "Current total process time: %f minutes" % ((currentTime - self.start_time) / 60.0)
            sys.stdout.flush()
            self.status_time = time.time()
            self.num_last_check = self.num
