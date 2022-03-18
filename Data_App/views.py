from fileinput import filename
from multiprocessing.dummy import Array
from turtle import towards
from django.shortcuts import render, HttpResponse, redirect
import os
from django.core.files.storage import FileSystemStorage
import itertools
import matplotlib.gridspec as gridspec
from parso import parse
from sklearn.cluster import *
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from astropy.io import fits
from astropy.io import ascii
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy import interpolate
import numpy as np
import pandas as pd
import time
import os
import glob
import warnings
import pyfiglet
import pickle
import csv

from django.http import StreamingHttpResponse



# /////////////////////////////////////////////////////////////////////////////

# def home(request):

#     # dir = 'Data_Handling/csv_file'
#     # for f in os.listdir(dir):
#     #     os.remove(os.path.join(dir, f))
#     return render(request, 'result.html')
#     return render(request, 'home.html')

def home(request):
    # path = os.getcwd()
    # final_path = os.path.join(path + '/Data_App/lc_files')
    # files = glob.glob(final_path)
    # for f in os.listdir(files[0]):
    #     os.remove(os.path.join(files[0],f))

    

    if request.method == 'POST':
        g =1
        for f in request.FILES:
            s = "doc" + str(g)
            g += 1
            file = request.FILES[s]
            if file == None:
                break
            fs = FileSystemStorage()
            n = fs.save(file.name, file)
            url = fs.url(n)
        return redirect('result')
    
    return render(request, 'result.html')
    # return render(request, 'adddata.html')





def uploadfiles(request):
    if request.method == 'POST':
        g =1
        for f in request.FILES:
            s = "doc" + str(g)
            g += 1
            file = request.FILES[s]
            if file == None:
                break
            fs = FileSystemStorage()
            n = fs.save(file.name, file)
            url = fs.url(n)
        return redirect('result')
    return HttpResponse("Bad Gateway")




def i(request):
   
    print('--------------------------------------------------------------------------------------')
    art = pyfiglet.figlet_format('       XSMWAS', font='epic')
    print(art)
    print('XSMWAS: Light Curve Waveform Analysis Software for Chandrayaan-II Solar X-ray Monitor')
    print('                                XSMWAS Version: 1.0')
    print('--------------------------------------------------------------------------------------\n\n')
    e=[]

    #Recording time of execution
    startCode = time.time()

    #random forest regressor fitting
    def rfr(time, data, nEst, dt):
        rf = RandomForestRegressor(n_estimators = nEst, random_state = 42)
        rf.fit(time.reshape(-1,1), data);
        time = np.arange(time.min(), time.max(), dt.min()) #sampling
        dataReg = rf.predict(time.reshape(-1,1))
        return (time, dataReg)

            
    #Defining the elementary flare profile function
    def efp(x, a, b, c, d, e):
        z = (b/c)+(c*d/2)
        return 0.886*a*c*np.exp(d*(b-x)+(c*d/2)**2)*(erf(z)-erf(z-(x/c)))+e

    #The feature extractor function
    def featureExtractor(filename, fig_path, csv_path, pkl_path):

    #Separating filename and extension
        filename, extension = filename.split('.')

        if (extension == 'fits') or (extension == 'lc'):	   	    
            #Reading the light curve from fits file
            lc = fits.open(os.path.join(final_path + '/' + file))
            data1 = lc[1].data	    
            #Reading in flux values and sample durations
            time1 = data1['TIME']
            data = data1['RATE']
            dt = time1[1:]-time1[:-1]
        
        elif (extension == 'txt') or (extension == 'dat'):	   	    
            #Reading the light curve from ascii table
            data1 = ascii.read(os.path.join(final_path + '/' + file))
            #Reading in flux values and sample durations
            time1 = data1['TIME']
            data = data1['RATE']
            dt = time1[1:]-time1[:-1]

        else:
            print('Please save with an appropriate extension.')
            return None

        #     #Noisy data figure
        #     plt.figure()
        #     plt.scatter(time1, data, s=1)
        #     plt.title(filename)

        #Smoothening individual streches of continuous data
        idx  = [i+1 for i,val in enumerate(dt) if val>100]
        idx.insert(0, 0)
        idx.append(len(time1))

        timeSmooth = [] #Time axis for smoothened data
        dataSmooth = [] #Smoothened data
        err = [] #Noise floor

        minWindow = (np.array(idx[1:])-np.array(idx[:-1])).min() #minwindow should be odd
        if minWindow%2==0:
            minWindow = minWindow-1
        else:
            minWindow = minWindow-2

        for i in range(len(idx)-1):
            timeTemp = time1[idx[i]:idx[i+1]]
            dataTemp = data[idx[i]:idx[i+1]]
            g = interpolate.interp1d(timeTemp, dataTemp, kind = 'cubic')

            #doing uniform sampling for savgol filter
            timeTempNew = np.arange(timeTemp.min(), timeTemp.max(), dt.min()) 
            dataTempNew = g(timeTempNew)

            #savgol filter for smoothening individual tracts of data
            #savgol filter assumes uniform sampling

            try: 
                dataSmoothTemp = savgol_filter(dataTempNew, minWindow, 3)
                errTemp = dataTempNew-dataSmoothTemp
            except ValueError:
                dataSmoothTemp = dataTempNew
                errTemp = dataTempNew-dataSmoothTemp


            #Downsampling for statistical noise removal
            f = interpolate.interp1d(timeTempNew, dataSmoothTemp, kind = 'cubic')
            timeTempNew = np.arange(timeTempNew.min(), timeTempNew.max(), 80*dt.min()) #sampling
            dataSmoothTemp = f(timeTempNew)

            #Non linear regression to remove spurious local maxima
            try:
                timeRegress, dataRegress = rfr(timeTempNew, dataSmoothTemp, 5, dt.min())
            except ValueError:
                timeRegress = timeTempNew
                dataRegress = dataSmoothTemp

            windowSav = len(dataRegress)//3
            if windowSav%2==0:
                windowSav = windowSav-1
            else:
                windowSav = windowSav-2

            try:     
                dataRegress = savgol_filter(dataRegress, windowSav, 3)
            except ValueError:
                dataRegress = dataRegress


        #         plt.figure()
        #         plt.scatter(timeTempNew, dataSmoothTemp)
        #         plt.plot(timeRegress, dataRegress, c = 'r')

            #Appending the data to the complete stretch
            dataSmooth.extend(dataRegress)
            timeSmooth.extend(timeRegress)
            err.extend(errTemp)

        timeSmooth = np.array(timeSmooth) #Saving as numpy array for convenience
        sigma = np.std(err) #Finding the deviation of noisefloor

        #Smoothened data figure
    #     plt.figure()
    #     plt.scatter(timeSmooth, dataSmooth, s=1)
    #     plt.title(filename)

        #upsampling for ease of locating local maxima - candidate flare events
        f1 = interpolate.interp1d(timeSmooth, dataSmooth, kind = 'cubic') #
        timenew = np.arange(timeSmooth.min(), timeSmooth.max(), dt.min()) 
        datanew = f1(timenew)


        #Finding extrema candidates
        idxExtrema = argrelextrema(datanew, np.greater, order = 20) #Emperically determine...


        #Finding total stretch of a single flare
        idxSearch = np.concatenate([[0], idxExtrema[0], [len(timenew)]])
        idxMinima = []

        #Generating array of two adjacent minima to maximas and adjust maxima indices
        for i in range(len(idxSearch)-1):
            dataTempShort = datanew[idxSearch[i]:idxSearch[i+1]]
            argMin = idxSearch[i]+np.where(dataTempShort == np.amin(dataTempShort))[0][0]
        
            
            #Checking for a different local maxima
            argAnomaly = idxSearch[i]+np.where(dataTempShort == np.amax(dataTempShort))[0][0]
            if (argAnomaly>argMin) and (argAnomaly!=idxSearch[i+1]):
                idxSearch[i+1]=argAnomaly
            elif (argAnomaly<argMin) and (argAnomaly!=idxSearch[i]):
                idxSearch[i]=argAnomaly 
                
            #Appending the local minima
            dataTempShort = datanew[idxSearch[i]:idxSearch[i+1]]
            argMin = idxSearch[i]+np.where(dataTempShort == np.amin(dataTempShort))[0][0]    
            idxMinima.append(argMin)
        e.append(filename)
                
        #Arrays for saving extracted features
        peakTimeArr = []
        peakFluxArr = []
        delPeakFluxArr = []
        decayRateArr = []
        startTimeArr = []
        endTimeArr = []
        riseTimeArr = []
        decayTimeArr = []
        durationArr = []
        fileNameArr = []
        categoryArr = []
        idxCandidate = []

        with open(pkl_path, 'rb') as file1:
            pickle_model = pickle.load(file1)  #Loading the ML model

        #Saving the extracted features to relevant arrays
        for i in range(len(idxMinima)-1):
            #Stretch of time and data
            timefit = timenew[idxMinima[i]:idxMinima[i+1]]
            datafit = datanew[idxMinima[i]:idxMinima[i+1]]
            
    #         plt.figure()
    #         plt.plot(timefit, datafit)
            
            #Heuristic way of guessing parameters
            aG = 1
            bG = timenew[idxExtrema[0][i]]
            cG = datanew[idxExtrema[0][i]]/2
            dG = 4/(timenew[idxMinima[i+1]]-timenew[idxMinima[i]])
            eG = min(datanew[idxMinima[i+1]], datanew[idxMinima[i]])

            #Curve fitting and finding optimal parameters
            try:
                pOpt, pCov = curve_fit(efp, timefit, datafit, p0 = [aG, bG, cG, dG, eG])
            except RuntimeError:
                pOpt = [0,0,1,0,0]

            #finding the preflare background
            bkg = pOpt[4]+0.1*sigma

            #time axis where the start and end time should be found
            timeAx = np.arange(bG-40000,bG+40000, dt.min())

            #calculating fitted curve and finding the start and stop time indices
            fit = efp(timeAx, *pOpt)
            idxSS = np.argwhere(np.diff(np.sign(fit - bkg))).flatten()


            #outlier detecttion
            #classification

            if len(idxSS)==2: #Discounting impossible flare events
                #finding the relevant parameters of interest

                peakTime = timenew[idxExtrema[0][i]]
                peakFlux = datanew[idxExtrema[0][i]]
                delPeakFlux = peakFlux - bkg
                decayRate = pOpt[3]
                startTime = timeAx[idxSS[0]]
                endTime = timeAx[idxSS[1]]
                riseTime = peakTime - startTime
                decayTime = endTime - peakTime
                duration = endTime - startTime
                features = np.array([peakTime, peakFlux, delPeakFlux, decayRate, startTime, endTime, riseTime, decayTime, duration])
                final_features = np.array([peakFlux, delPeakFlux, decayRate, riseTime, decayTime])
                flag = np.any((features<0))
                

                #Appending the parameters to the relevant arrays -- removing outliers
                if ((delPeakFlux>=sigma) and ~flag):
                    idxCandidate.append(idxExtrema[0][i]) #index of suitable flare
                    peakTimeArr.append(peakTime)
                    peakFluxArr.append(peakFlux)
                    delPeakFluxArr.append(delPeakFlux)
                    decayRateArr.append(decayRate)
                    startTimeArr.append(startTime)
                    endTimeArr.append(endTime)
                    riseTimeArr.append(riseTime)
                    decayTimeArr.append(decayTime)
                    durationArr.append(duration)
                    fileNameArr.append(filename)
                    categoryArr.append(pickle_model.predict(final_features.reshape(1,-1)))
                    

                #figures if needed can be generated and saved accordingly

    #             plt.figure()
    #             plt.plot(timefit, datafit)
    #             plt.plot(timeAx, fit)
    #             plt.plot(timeAx, bkg*np.ones(len(timeAx)))
    #             plt.scatter(timeAx[idxSS], fit[idxSS], c = 'g', marker = 'x')
    #             plt.xlabel('Time (s)')
    #             plt.ylabel('Flux (counts/s)')
    #             plt.scatter(peakTime, peakFlux, c= 'r')




# *************************************************************************************************
        toWrite = pd.concat([pd.Series(fileNameArr, name='File Name'),
                            pd.Series(peakTimeArr,name='Peak Time (s)'),
                            pd.Series(peakFluxArr,name='Peak Flux (counts/s)'), 
                            pd.Series(delPeakFluxArr,name='Peak Above bkg (counts/s)'), 
                            pd.Series(decayRateArr,name='Decay Rate (/s)'), 
                            pd.Series(startTimeArr,name='Start Time (s)'), 
                            pd.Series(endTimeArr,name='End Time (s)'), 
                            pd.Series(riseTimeArr,name='Rise Time (s)'), 
                            pd.Series(decayTimeArr,name='Decay Time (s)'), 
                            pd.Series(durationArr,name='Duration (s)'),
                            pd.Series(categoryArr, name = 'Category of Flare')], axis=1)
        toWrite.to_csv (os.path.join(csv_path + filename + '.csv'), index = False)
# *************************************************************************************************




        # print(riseTimeArr)
        print('Features extracted and saved ...')

        #Plotting the results
        plt.figure(dpi=300)
        plt.scatter(time1, data, s = 1)
        plt.plot(timenew, datanew, c= 'r')
        plt.scatter(timenew[idxCandidate], datanew[idxCandidate], c='g', marker='x')
        plt.xlabel('Time (s)')
        plt.ylabel('Flux (counts/s)')
        plt.title(filename)
        plt.legend(['Processed light curve', 'Raw data', 'Detected flare peaks'])
        if saveImgFlag:
            plt.savefig(os.path.join(fig_path + filename+'.jpg'))
            print('Light curve plot saved ...')
        if viewImgFlag:
            plt.show()
        #plt.close()
        return None


    warnings.filterwarnings("ignore")
    path = os.getcwd()
    final_path = os.path.join(path + '/Data_App/lc_files')
    names = os.listdir(final_path)
    print(f"No. of Files Detected: {len(names)} \n")
    count = 0

    if not os.path.isdir('fig_files'):
        os.mkdir('fig_files')

    if not os.path.isdir('csv_files'):
        os.mkdir('csv_files')
    
    # imgSave = input('Save images? (y/n)\n')
    # imgView = input('Interactively view images? (y/n)\n')

    saveImgFlag = 1
    # if imgSave == 'y':
    #     saveImgFlag = 1

    viewImgFlag = 0
    # if imgView == 'y':
    #     viewImgFlag = 1 
        
    for file in names:
        start_time = time.time()
        count += 1
        print(f'\nExtracting Features from file: {count}')
        fig_path = os.path.join(path + r'/Data_App/lc_files/')
        csv_path = os.path.join(path + r'/Data_App/lc_files/')
        pkl_path = os.path.join(path + r'/Data_App/xsm_model.pkl')
        final_data = featureExtractor(file, fig_path, csv_path, pkl_path)
        print(f'Total Time taken: {time.time() - start_time} seconds \n')
        
    print(f'\nDone! Finished in {time.time() - startCode} seconds\n')
    print('--------------------------------------------------------------------------------------')

    return render(request, 'result.html',{"name":e})

def run():
    return HttpResponse("Gg")

    
    