Extra credit: added Alaska and Hawaii in the map
      	      plotted the ROC curve for both classifers
Use of slip days: 4
* in reddit_model.py:
  + we read from parquet files for both comments-minimal and submissions
  + the fraction of samples that we choose is stored in variable called 'sampleRate'
    and is set to 0.01 in our code at line 15
  + the code to plot ROC curve is in reddit_model.py, right after TASK 7
    it used sklearn library, which needs to be installed on the vm
