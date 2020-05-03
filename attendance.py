import pandas as pd
import config
import numpy as np
import os
import datetime

class AttendanceMarker:

    def __init__(self):
        # current datetime to put attendance 
        now = datetime.datetime.now()
        self.time = now.strftime(config.DATE_TIME_FORMAT)

    def _create_new_csv(self):
        names = os.listdir(config.FACE_DATABASE_DIR)
        names = np.array(names)
        df = pd.DataFrame(data=names,columns=[config.CSV_COL_NAME])
        df.to_csv(config.ATTENDANCE_FILENAME,index=False)



    def mark_attendance(self,names):

        if(not os.path.exists(config.ATTENDANCE_FILENAME)):
            self._create_new_csv()

        df = pd.read_csv(config.ATTENDANCE_FILENAME)
        df[self.time] = 0
        for name in names:
            df.loc[df[config.CSV_COL_NAME] == name,self.time] = 1

        df.to_csv(config.ATTENDANCE_FILENAME,index=False)
        print('Saving attendance for  names -> [ {} ] to file :{} at : {}'.format(names, config.ATTENDANCE_FILENAME,self.time))



