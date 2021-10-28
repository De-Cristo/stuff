import random
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from root_numpy import root2array, rec2array, array2root
from skTMVA import convert_bdt_sklearn_tmva
import sys
import os
import awkward as ak
import ROOT as R
from array import *

def createDir(OutDir):
    if OutDir is not None :
        if not os.path.isdir( OutDir ) :
            os.makedirs( OutDir )

# ------------------------- plotting Tools -------------------------



# ------------------ do reweight BDT validation --------------------

def extract_reWeight_Dovalidation(filestring, X_test_rew, BDTDecision, y_test_rew):

    #first let's read data
    print("============ starting comparePassFail ===========")
    Xstack = np.hstack((X_test_rew,BDTDecision.reshape(BDTDecision.shape[0], -1),y_test_rew.reshape(y_test_rew.shape[0], -1)))
    print("============ find best binning for bkg pass the selection ===========")
    tuple_bkgpass = [d[0:-1] for d in Xstack if d[12]==1]
    tuple_bkgfail = [d[0:-1] for d in Xstack if d[12]==0]
    print('tuple_bkgpass ',np.shape(tuple_bkgpass))
    print('tuple_bkgfail ',np.shape(tuple_bkgfail))
    low = min(np.min(d[11]) for d in Xstack if d[12]==1)
    high = max(np.max(d[11]) for d in Xstack if d[12]==1)
    if low<-1 or high>1:
        print("error! BDT not in (-1,1)")
        exit()
    #low_high = (low,high)
    #print("low_high",low_high)
    #low=-1
    #high=1
    decisions = [d[-2] for d in Xstack if d[12]==1]
    # --------------------------------------   get flat binning
    s_tot = np.ones(len(decisions)).sum()
    nbins = 30
    values = np.sort(decisions)
    cumu = np.cumsum( np.ones(len(decisions))  )
    targets = [n*s_tot/nbins for n in range(1,nbins)]
    workingPoints = []
    for target in targets:
        index = np.argmax(cumu>target)
        workingPoints.append(values[index])
    bbins_comparePassFail = [float(low)]+workingPoints+[float(high)]
    print(bbins_comparePassFail)

    #third overlay passed bkg and failed bkg
    histspass = R.TH1F("histspass","bkg pass",nbins,-1,1)
    histsfail = R.TH1F("histsfail","bkg fail",nbins,-1,1)
    histsweight = R.TH1F("histsweight","weight",nbins,-1,1)
    runArray = array('d',bbins_comparePassFail)
    histspass.SetBins(len(runArray)-1, runArray)
    histsfail.SetBins(len(runArray)-1, runArray)
    histsweight.SetBins(len(runArray)-1, runArray)
    for i in range(np.shape(tuple_bkgpass)[0]):
        histspass.Fill(tuple_bkgpass[i][11],tuple_bkgpass[i][10])
    for i in range(np.shape(tuple_bkgfail)[0]):
        histsfail.Fill(tuple_bkgfail[i][11],tuple_bkgfail[i][10])
    #calculate the weight for each bin 
    for i in range(1,nbins+1):
        histsweight.SetBinContent(i, histspass.GetBinContent(i)/histsfail.GetBinContent(i))
    fout = R.TFile.Open(OutDir+filestring,'RECREATE')
    fout.cd()
    histspass.Write()
    histsfail.Write()
    histsweight.Write()
    # ----------------   Reweight inputs -----------
    binset = {
             'selLeptons_pt_0':[60, 0, 200],
             'VHHFatJet1_Pt':[60,200,800],
             'VHHFatJet2_Pt':[60,200,600],
             'VHHFatJet1_Msoftdrop':[60,50,250],
             'VHHFatJet2_Msoftdrop':[60,50,200],
             'VHHFatJet_mjj':[60,0,2000],
             'VHHFatJet_HHPt':[60,0,1000],
             #'selLeptons_phi_0':[50,-3.14,3.14],
             #'VHHFatJet1_phi':[50,-3.14,3.14],
             #'VHHFatJet2_phi':[50,-3.14,3.14],
    }
    order={'selLeptons_pt_0':0,'VHHFatJet1_Pt':1,'VHHFatJet2_Pt':2,'VHHFatJet1_Msoftdrop':3,'VHHFatJet2_Msoftdrop':4,'VHHFatJet_mjj':5,'VHHFatJet_HHPt':6,'selLeptons_phi_0':7,'VHHFatJet1_phi':8,'VHHFatJet2_phi':9}
    # ---------- make 2D plots --------------
    reweighted_hist={}
    reweighted_hist2={}
    histspass={}
    histsfail={}
    histsfail_rew={}
    for iplot in binset.keys():
        reweighted_hist[iplot] = R.TH2F(iplot+'reweighted_hist',iplot+'reweighted_hist',len(runArray)-1, runArray,binset[iplot][0],binset[iplot][1],binset[iplot][2])
        reweighted_hist2[iplot] = R.TH2F(iplot+'reweighted_hist2',iplot+'reweighted_hist2',len(runArray)-1, runArray,binset[iplot][0],binset[iplot][1],binset[iplot][2])

        histspass[iplot] = R.TH1F(iplot+'pass',iplot+'pass',binset[iplot][0],binset[iplot][1],binset[iplot][2])
        for i in range(np.shape(tuple_bkgpass)[0]):
            histspass[iplot].Fill(tuple_bkgpass[i][order[iplot]],tuple_bkgpass[i][10])
        histspass[iplot].Scale(1.0/ histspass[iplot].Integral())
        histspass[iplot].SetTitle(iplot+'_pass')
        histspass[iplot].SetName(iplot+'_pass')
        for i in range(np.shape(tuple_bkgfail)[0]):
            reweighted_hist[iplot].Fill(tuple_bkgfail[i][11],tuple_bkgfail[i][order[iplot]],tuple_bkgfail[i][10])
        histsfail[iplot] = reweighted_hist[iplot].ProjectionY()
        for ibx in range(1,reweighted_hist[iplot].GetNbinsX()+1):
            for iby in range(1,reweighted_hist[iplot].GetNbinsY()+1):
                reweighted_hist2[iplot].SetBinContent(ibx,iby,reweighted_hist[iplot].GetBinContent(ibx,iby)*histsweight.GetBinContent(ibx))
                reweighted_hist2[iplot].SetBinError(ibx,iby,reweighted_hist[iplot].GetBinError(ibx,iby)*histsweight.GetBinContent(ibx))
        print('iplot',iplot)
        histsfail_rew[iplot] = reweighted_hist2[iplot].ProjectionY()
        print('histsfail[iplot].Integral()',histsfail[iplot].Integral(),'histsfail_rew[iplot].Integral()',histsfail_rew[iplot].Integral())
        histsfail[iplot].Scale(1.0/histsfail[iplot].Integral())
        histsfail[iplot].SetTitle(iplot+'_fail')
        histsfail[iplot].SetName(iplot+'_fail')
        histsfail_rew[iplot].Scale(1.0/histsfail_rew[iplot].Integral())
        histsfail_rew[iplot].SetTitle(iplot+'_fail_rew')
        histsfail_rew[iplot].SetName(iplot+'_fail_rew')
    fout.cd()
    for iplot in binset.keys():
        histspass[iplot].Write()
        histsfail[iplot].Write()
        histsfail_rew[iplot].Write()

def plot_BDTreweight(VHHfiles, procs, addtional):
    legname=['Pass','Fail','FailReweight']
    for ifile in VHHfiles:
        hsample = {}
        fin = R.TFile.Open('{0}'.format(ifile),'READ')
        #Get all plots
        MAX_ = 0
        MaxX_ = 0
        MinX_ = 0
        NX_ = 0
        Min_ = 0
        for iprocs in procs:
            fin.cd()
            htmp = fin.Get(iprocs).Clone()
            htmp.Scale(1.0/htmp.Integral())
            NX_=htmp.GetNbinsX()
            MinX_=htmp.GetBinCenter(1) - htmp.GetBinWidth(1)/2.0
            MaxX_=htmp.GetBinCenter(NX_) + htmp.GetBinWidth(NX_)/2.0
            MAX_ = htmp.GetMaximum() if htmp.GetMaximum() > MAX_ else MAX_
            Min_ = htmp.GetMinimum() if htmp.GetMinimum() < Min_ else Min_
            hsample[iprocs] = R.TH1F(iprocs,iprocs,NX_,1,1+NX_)
            for i in range(1,NX_+1):
                hsample[iprocs].SetBinContent(i,htmp.GetBinContent(i))
                hsample[iprocs].SetBinError(i,htmp.GetBinError(i))
        print(NX_,MinX_,MaxX_)
        #make plots
        c=R.TCanvas()
        c.SetFillColor(0)
        c.SetBorderMode(0)
        c.SetBorderSize(2)
        c.SetFrameBorderMode(0)
        frame_4fa51a0__1 = R.TH1D("frame_4fa51a0__1","",NX_,1,1+NX_)
        frame_4fa51a0__1.GetXaxis().SetTitle("transformed BDT variable")
        frame_4fa51a0__1.GetXaxis().SetTitleSize(0.05)
        frame_4fa51a0__1.GetXaxis().SetTitleOffset(0.8)
        frame_4fa51a0__1.GetXaxis().SetTitleFont(42)
        frame_4fa51a0__1.GetYaxis().SetTitle("")
        frame_4fa51a0__1.GetYaxis().SetLabelFont(42)
        frame_4fa51a0__1.GetYaxis().SetLabelSize(0.05)
        frame_4fa51a0__1.GetYaxis().SetTitleSize(0.05)
        frame_4fa51a0__1.GetYaxis().SetTitleFont(42)
        frame_4fa51a0__1.GetYaxis().SetRangeUser(Min_*0.7,MAX_*1.3)
        frame_4fa51a0__1.Draw("AXISSAME")
        R.gStyle.SetOptStat(0)
        #c.SetLogy()
        leg = R.TLegend(0.6,0.6,0.85,0.8)
        leg.SetBorderSize(0)
        leg.SetLineStyle(1)
        leg.SetLineWidth(1)
        i=0
        for iprocs in procs:
            print(iprocs)
            hsample[iprocs].SetLineColor(i+2)
            hsample[iprocs].SetLineWidth(2)
            hsample[iprocs].SetMarkerColor(i+2)
            hsample[iprocs].Draw("same HIST")
            entry=leg.AddEntry(iprocs,legname[i] ,"lp")
            entry.SetFillStyle(1001)
            entry.SetMarkerStyle(8)
            entry.SetMarkerSize(1.5)
            entry.SetLineStyle(1)
            entry.SetLineWidth(3)
            entry.SetTextFont(42)
            entry.SetTextSize(0.06)
            i+=1
        leg.Draw()
        tex = R.TLatex(0.14,0.95,"CMS")
        tex.SetNDC()
        tex.SetTextAlign(13)
        tex.SetTextSize(0.048)
        tex.SetLineWidth(2)
        tex.Draw()
        tex1 = R.TLatex(0.22,0.95,"Simulation Work in Progress")
        tex1.SetNDC()
        tex1.SetTextAlign(13)
        tex1.SetTextFont(52)
        tex1.SetTextSize(0.03648)
        tex1.SetLineWidth(2)
        tex1.Draw()
        c.SaveAs("%s_weight%s.png"%(ifile[0:-5], addtional) )
        c.SaveAs("%s_weight%s.pdf"%(ifile[0:-5],addtional) )
        c.SaveAs("%s_weight%s.C"%(ifile[0:-5],addtional) )

def plot_reweighted_inputs(VHHfiles,procs, addtional):
    for ifile in VHHfiles:
        hsample = {}
        fin = R.TFile.Open('{0}'.format(ifile),'READ')
        #Get all plots
        MAX_ = 0
        MaxX_ = 0
        MinX_ = 0
        NX_ = 0
        Min_ = 0
        for iprocs in procs:
            for iprocs_v in [iprocs+'_pass',iprocs+'_fail',iprocs+'_fail_rew']:
                hsample[iprocs_v] = fin.Get(iprocs_v).Clone()
                hsample[iprocs_v].Scale(1.0/hsample[iprocs_v].Integral())
                MAX_ = hsample[iprocs_v].GetMaximum() if hsample[iprocs_v].GetMaximum() > MAX_ else MAX_
                Min_ = hsample[iprocs_v].GetMinimum() if hsample[iprocs_v].GetMinimum() < Min_ else Min_
                NX_=hsample[iprocs_v].GetNbinsX()
                MinX_=hsample[iprocs_v].GetBinCenter(1) - hsample[iprocs_v].GetBinWidth(1)/2.0
                MaxX_=hsample[iprocs_v].GetBinCenter(NX_) + hsample[iprocs_v].GetBinWidth(NX_)/2.0

            #make plots
            c=R.TCanvas()
            c.SetFillColor(0)
            c.SetBorderMode(0)
            c.SetBorderSize(2)
            c.SetFrameBorderMode(0)
            frame_4fa51a0__1 = R.TH1D("frame_4fa51a0__1","",NX_,MinX_,MaxX_)
            frame_4fa51a0__1.GetXaxis().SetTitle(iprocs)
            frame_4fa51a0__1.GetXaxis().SetTitleSize(0.05)
            frame_4fa51a0__1.GetXaxis().SetTitleOffset(0.8)
            frame_4fa51a0__1.GetXaxis().SetTitleFont(42)
            frame_4fa51a0__1.GetYaxis().SetTitle("")
            frame_4fa51a0__1.GetYaxis().SetLabelFont(42)
            frame_4fa51a0__1.GetYaxis().SetLabelSize(0.05)
            frame_4fa51a0__1.GetYaxis().SetTitleSize(0.05)
            frame_4fa51a0__1.GetYaxis().SetTitleFont(42)
            frame_4fa51a0__1.GetYaxis().SetRangeUser(Min_*0.7,MAX_*1.1)
            frame_4fa51a0__1.Draw("AXISSAME")
            R.gStyle.SetOptStat(0)
            #c.SetLogy()
            leg = R.TLegend(0.6,0.5,0.8,0.8)
            leg.SetBorderSize(0)
            leg.SetLineStyle(1)
            leg.SetLineWidth(1)
            i=0
            legname=['Pass','Fail','FailReweight']
            for iprocs_v in [iprocs+'_pass',iprocs+'_fail',iprocs+'_fail_rew']:
                print(iprocs_v)
                hsample[iprocs_v].SetMarkerColor(i+2)
                hsample[iprocs_v].SetLineColor(i+2)
                hsample[iprocs_v].SetLineWidth(2)
                if i<2:
                    hsample[iprocs_v].Draw("same HIST")
                else:
                    hsample[iprocs_v].Draw("same e")
                entry=leg.AddEntry(iprocs_v,legname[i] ,"lp")
                entry.SetFillStyle(1001)
                entry.SetMarkerStyle(9)
                entry.SetMarkerSize(1.5)
                entry.SetLineStyle(1)
                entry.SetLineWidth(3)
                entry.SetTextFont(42)
                entry.SetTextSize(0.06)
                i+=1
            leg.Draw()
            tex = R.TLatex(0.14,0.95,"CMS")
            tex.SetNDC()
            tex.SetTextAlign(13)
            tex.SetTextSize(0.048)
            tex.SetLineWidth(2)
            tex.Draw()
            tex1 = R.TLatex(0.22,0.95,"Simulation Work in Progress")
            tex1.SetNDC()
            tex1.SetTextAlign(13)
            tex1.SetTextFont(52)
            tex1.SetTextSize(0.03648)
            tex1.SetLineWidth(2)
            tex1.Draw()

            c.SaveAs("%s%s_%s.png"%(ifile[0:-5], iprocs,addtional) )
            c.SaveAs("%s%s_%s.pdf"%(ifile[0:-5],iprocs,addtional) )
            c.SaveAs("%s%s_%s.C"%(ifile[0:-5],iprocs,addtional) )

# ------------------------------ end -------------------------------


def flatten_bdt_output(clf, X_train, y_train, X_test, y_test, outputFilename):
    '''compare train and test histogram with flatten signal'''
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    print("low_high",low_high)

    # --------------------------------------   get flat binning 
    s_tot = np.ones(len(decisions[0])).sum()
    if s_tot>15:
        bins = 30
    elif s_tot<5:
        bins = 5
    else:
        bins = int(s_tot)
    bins = 15
    #print("Signal total: ",s_tot)
    values = np.sort(decisions[0])
    #print('values',values)
    cumu = np.cumsum( np.ones(len(decisions[0]))  )
    #print('cumu',cumu)
    targets = [n*s_tot/bins for n in range(1,bins)]
    #print('targets',targets)
    workingPoints = []
    for target in targets:
        index = np.argmax(cumu>target)
        workingPoints.append(values[index])
    bins = [float(low)]+workingPoints+[float(high)]

    plt.figure()
    fig, ax = plt.subplots(figsize=(6,5))
    plt.xlim([-1.05, 1.05])
    countsS, bbins, bars =plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=False,
             label='S (train)')
    print(countsS, bbins, bars)
    countsB, bbins, bars = plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=False,
             label='B (train)')
    print(countsB, bbins, bars)
    
    weight = np.ones_like(decisions[2])*len(decisions[0])/len(decisions[2])
    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=False, weights=weight)
    scale = len(decisions[0]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    weight = np.ones_like(decisions[3])*len(decisions[0])/len(decisions[2])
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=False, weights=weight)
    scale = len(decisions[0]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.savefig(OutDir+outputFilename+'bdt_flatten.png')
 
    #Plot S/sqrt(B)
    plt.figure(1)
    svsb = countsS/np.sqrt(countsB)
    #print(center.shape)
    #print(svsb.shape)
    fig, ax = plt.subplots()
    plt.plot(center, svsb,lw=1, color='b', label='sum %0.2f'%(np.sqrt((svsb**2).sum())))
    plt.xlabel('BDT output')
    plt.ylabel('S/sqrt(B)')
    plt.grid()
    plt.savefig(OutDir+outputFilename+"bdt_flatten_svsb.png")
    np.savez(OutDir+outputFilename+'bdt_flatten.npz', center=center, decisions_0=decisions[0], decisions_1=decisions[1], decisions_2=decisions[2], decisions_3=decisions[3], svsb=svsb )


def makedatacards_norew(outputFilename,passcut, flattencut,procs,infilename,branchlist,bdt):
    print("============ Reading files and make datacards ===========")
    fout = R.TFile.Open(outputFilename,'RECREATE')
    reweighted_hist = {}
    reweighted_hist2 = {}
    hists = {}
    print("============ Flatten SvB  ===========")
    bbins_svbflatten = []
    if len(bbins_svbflatten) == 0:
        print("============ find best binning ===========")
        signalinfilename = 'Boosted/*CV*root'
        signal = root2array(signalinfilename, "Events", branches=branchlist, selection=flattencut)
        signal = rec2array(signal)
        BDT_out = bdt.decision_function(pd.DataFrame(signal))
        decisions = BDT_out
        low = min(np.min(d) for d in decisions)
        high = max(np.max(d) for d in decisions)
        low_high = (low,high)
        print("low_high",low_high)
        low=-1
        high=1
        # --------------------------------------   get flat binning
        s_tot = np.ones(len(decisions)).sum()
        bins = 30
        values = np.sort(decisions)
        cumu = np.cumsum( np.ones(len(decisions))  )
        targets = [n*s_tot/bins for n in range(1,bins)]
        workingPoints = []
        for target in targets:
            index = np.argmax(cumu>target)
            workingPoints.append(values[index])
        bbins_svbflatten = [float(low)]+workingPoints+[float(high)]
        print(bbins_svbflatten)

    for ifile in procs:
        print(ifile)
        if ifile not in infilename.keys():
            hists[ifile] = R.TH1F(ifile,ifile,50,-1,1)
            runArray = array('d',bbins_svbflatten)
            hists[ifile].SetBins(len(runArray)-1, runArray)
            fout.cd()
            hists[ifile].Write()
            continue
        signal = root2array(infilename[ifile], "Events", branches=branchlist, selection=passcut)
        signal = rec2array(signal)
        signal_w = root2array(infilename[ifile], "Events", branches=[ 'weight'], selection=passcut)
        signal_w = rec2array(signal_w)
        BDT_out = bdt.decision_function(pd.DataFrame(signal))
        signal_w = signal_w.flatten('K')
        cat =np.array([BDT_out,signal_w])
        #create histogram with flatten binning: bbins_svbflatten
        hists[ifile] = R.TH1F(ifile,ifile,50,-1,1)
        runArray = array('d',bbins_svbflatten)
        hists[ifile].SetBins(len(runArray)-1, runArray)
        for i in range(cat.shape[1]):
            hists[ifile].Fill(cat[0][i],cat[1][i])
        hists[ifile].Scale(137/58.8)
        if 'TTB' in ifile:
            hists[ifile].Scale(1.32)
        hists[ifile].Write()


def makedatacards(outputFilename,passcut,failcut, flattencut,weightfile,weighthist,procs,infilename,branchlist,bdt,bdt_reweight,do_first):
    fin = R.TFile.Open('{0}'.format(weightfile),'READ')
    MaxX_BDT = []
    print("try to find ",weighthist, " in ", weightfile)
    hrew = fin.Get(weighthist).Clone()
    for i in range(1,hrew.GetNbinsX()+2):
        MaxX_BDT.append(hrew.GetBinLowEdge(i))
    print('rew:',len(MaxX_BDT),MaxX_BDT)
    runArray_rew = array('d',MaxX_BDT)
    print("============ Reading files and make datacards ===========")
    fout = R.TFile.Open(outputFilename,'RECREATE')
    reweighted_hist = {}
    reweighted_hist2 = {}
    hists = {}
    print("============ Flatten SvB  ===========")
    bbins_svbflatten = []
    if len(bbins_svbflatten) == 0:
        print("============ find best binning ===========")
        signalinfilename = 'Boosted/*CV*root'
        signal = root2array(signalinfilename, "Events", branches=branchlist, selection=flattencut)
        signal = rec2array(signal)
        BDT_out = bdt.decision_function(pd.DataFrame(signal))
        decisions = BDT_out
        low = min(np.min(d) for d in decisions)
        high = max(np.max(d) for d in decisions)
        low_high = (low,high)
        print("low_high",low_high)
        low=-1
        high=1
        # --------------------------------------   get flat binning
        s_tot = np.ones(len(decisions)).sum()
        bins = 30
        values = np.sort(decisions)
        cumu = np.cumsum( np.ones(len(decisions))  )
        targets = [n*s_tot/bins for n in range(1,bins)]
        workingPoints = []
        for target in targets:
            index = np.argmax(cumu>target)
            workingPoints.append(values[index])
        bbins_svbflatten = [float(low)]+workingPoints+[float(high)]
        print(bbins_svbflatten)

    for ifile in procs:
        print(ifile)
        if ifile not in infilename.keys():
            hists[ifile] = R.TH1F(ifile,ifile,50,-1,1)
            runArray = array('d',bbins_svbflatten)
            hists[ifile].SetBins(len(runArray)-1, runArray)
            fout.cd()
            hists[ifile].Write()
            continue
        signal = root2array(infilename[ifile], "Events", branches=branchlist, selection=passcut)
        signal = rec2array(signal)
        signal_w = root2array(infilename[ifile], "Events", branches=[ 'weight'], selection=passcut)
        signal_w = rec2array(signal_w)
        BDT_out = bdt.decision_function(pd.DataFrame(signal))
        signal_w = signal_w.flatten('K')
        cat =np.array([BDT_out,signal_w])
        #create histogram with flatten binning: bbins_svbflatten
        hists[ifile] = R.TH1F(ifile,ifile,50,-1,1)
        runArray = array('d',bbins_svbflatten)
        hists[ifile].SetBins(len(runArray)-1, runArray)
        for i in range(cat.shape[1]):
            hists[ifile].Fill(cat[0][i],cat[1][i])
        if 'CV' not in ifile:
            #bkg
            failbkg = root2array(infilename[ifile], "Events", branches=branchlist, selection=failcut)
            failbkg = rec2array(failbkg)
            failbkg_w = root2array(infilename[ifile], "Events", branches=[ 'weight'], selection=failcut)
            failbkg_w = rec2array(failbkg_w)
            BDT_outfail = bdt.decision_function(pd.DataFrame(failbkg))
            BDT_outfail_weight = bdt_reweight.decision_function(pd.DataFrame(failbkg))
            failbkg_w = failbkg_w.flatten('K')
            cat_rew =np.array([BDT_outfail_weight,BDT_outfail,failbkg_w])
            print('cat_rew.shape ',cat_rew.shape)
            print(cat_rew[:][0,int(len(cat_rew[0])/2)])
            if do_first:
                cat_rew = cat_rew[:,0:int(len(cat_rew[0])/2)]
            else:
                cat_rew = cat_rew[:,int(len(cat_rew[0])/2):-1]
            print('cat_rew.shape ',cat_rew.shape)
            reweighted_hist[ifile] =  R.TH2F(ifile+'reweighted_hist',ifile+'reweighted_hist',len(runArray_rew)-1, runArray_rew,len(runArray)-1, runArray)
            reweighted_hist[ifile+'_weight_Up'] =  R.TH2F(ifile+'reweighted_hist_weight_Up',ifile+'reweighted_hist_weight_Up',len(runArray_rew)-1, runArray_rew,len(runArray)-1, runArray)
            reweighted_hist[ifile+'_weight_Down'] =  R.TH2F(ifile+'reweighted_hist_weight_Down',ifile+'reweighted_hist_weight_Down',len(runArray_rew)-1, runArray_rew,len(runArray)-1, runArray)
            for i in range(cat_rew.shape[1]):
                reweighted_hist[ifile].Fill(cat_rew[0][i],cat_rew[1][i],cat_rew[2][i])
                reweighted_hist[ifile+'_weight_Up'].Fill(cat_rew[0][i],cat_rew[1][i],cat_rew[2][i])
                reweighted_hist[ifile+'_weight_Down'].Fill(cat_rew[0][i],cat_rew[1][i],cat_rew[2][i])
            for ibx in range(1,reweighted_hist[ifile].GetNbinsX()+1):
                for iby in range(1,reweighted_hist[ifile].GetNbinsY()+1):
                    #do reweighting
                    reweighted_hist[ifile].SetBinContent(ibx,iby,reweighted_hist[ifile].GetBinContent(ibx,iby)*hrew.GetBinContent(ibx))
                    reweighted_hist[ifile].SetBinError(ibx,iby,reweighted_hist[ifile].GetBinError(ibx,iby)*hrew.GetBinContent(ibx))
                    if ibx<reweighted_hist[ifile].GetNbinsX():
                        reweighted_hist[ifile+'_weight_Up'].SetBinContent(ibx,iby,reweighted_hist[ifile+'_weight_Up'].GetBinContent(ibx,iby)*hrew.GetBinContent(ibx+1))
                        reweighted_hist[ifile+'_weight_Up'].SetBinError(ibx,iby,reweighted_hist[ifile+'_weight_Up'].GetBinError(ibx,iby)*hrew.GetBinContent(ibx+1))
                    else:
                        reweighted_hist[ifile+'_weight_Up'].SetBinContent(ibx,iby,reweighted_hist[ifile+'_weight_Up'].GetBinContent(ibx,iby)*hrew.GetBinContent(ibx))
                        reweighted_hist[ifile+'_weight_Up'].SetBinError(ibx,iby,reweighted_hist[ifile+'_weight_Up'].GetBinError(ibx,iby)*hrew.GetBinContent(ibx))
                    if ibx>1:
                        reweighted_hist[ifile+'_weight_Down'].SetBinContent(ibx,iby,reweighted_hist[ifile+'_weight_Down'].GetBinContent(ibx,iby)*hrew.GetBinContent(ibx-1))
                        reweighted_hist[ifile+'_weight_Down'].SetBinError(ibx,iby,reweighted_hist[ifile+'_weight_Down'].GetBinError(ibx,iby)*hrew.GetBinContent(ibx-1))
                    else:
                        reweighted_hist[ifile+'_weight_Down'].SetBinContent(ibx,iby,reweighted_hist[ifile+'_weight_Down'].GetBinContent(ibx,iby)*hrew.GetBinContent(ibx))
                        reweighted_hist[ifile+'_weight_Down'].SetBinError(ibx,iby,reweighted_hist[ifile+'_weight_Down'].GetBinError(ibx,iby)*hrew.GetBinContent(ibx))

            hists[ifile+'_FailReweight'] = reweighted_hist[ifile].ProjectionY()
            hists[ifile+'_FailReweight'].SetTitle(ifile+'_FailReweight')
            hists[ifile+'_FailReweight'].SetName(ifile+'_FailReweight')
            hists[ifile+'_FailReweight_weight_Up'] = reweighted_hist[ifile+'_weight_Up'].ProjectionY()
            hists[ifile+'_FailReweight_weight_Up'].SetTitle(ifile+'_FailReweight_weight_Up')
            hists[ifile+'_FailReweight_weight_Up'].SetName(ifile+'_FailReweight_weight_Up')
            hists[ifile+'_FailReweight_weight_Down'] = reweighted_hist[ifile+'_weight_Down'].ProjectionY()
            hists[ifile+'_FailReweight_weight_Down'].SetTitle(ifile+'_FailReweight_weight_Down')
            hists[ifile+'_FailReweight_weight_Down'].SetName(ifile+'_FailReweight_weight_Down')
            if 'TTB' in ifile:
                hists[ifile].Scale(1.32*137/58.8)
                hists[ifile+'_FailReweight'].Scale(1.32*137/58.8)
                hists[ifile+'_FailReweight_weight_Up'].Scale(1.32*137/58.8)
                hists[ifile+'_FailReweight_weight_Down'].Scale(1.32*137/58.8)
            else:
                hists[ifile].Scale(137/58.8)
                hists[ifile+'_FailReweight'].Scale(137/58.8)
                hists[ifile+'_FailReweight_weight_Up'].Scale(137/58.8)
                hists[ifile+'_FailReweight_weight_Down'].Scale(137/58.8)
            #reweight the passed samples
            hists[ifile+'_PassedBKG'] = hists[ifile].Clone()
            npassbkg=hists[ifile].Integral()
            nfailbkg=hists[ifile+'_FailReweight'].Integral()

            hists[ifile].Scale( npassbkg/(nfailbkg+npassbkg) )
            hists[ifile+'_FailReweight'].Scale(npassbkg/(nfailbkg+npassbkg))            
            hists[ifile].Add(hists[ifile+'_FailReweight'])
            hists[ifile+'_FailReweight'].Scale(hists[ifile+'_PassedBKG'].Integral()/hists[ifile+'_FailReweight'].Integral())
            hists[ifile+'_FailReweight_weight_Up'].Scale(hists[ifile+'_PassedBKG'].Integral()/hists[ifile+'_FailReweight_weight_Up'].Integral())
            hists[ifile+'_FailReweight_weight_Down'].Scale(hists[ifile+'_PassedBKG'].Integral()/hists[ifile+'_FailReweight_weight_Down'].Integral())
            hists[ifile+'_PassedBKG'].SetTitle(ifile+'_PassedBKG')
            hists[ifile+'_PassedBKG'].SetName(ifile+'_PassedBKG')
            hists[ifile+'_PassedBKG'].Write()
            hists[ifile+'_FailReweight'].Write()
            hists[ifile+'_FailReweight_weight_Up'].Write()
            hists[ifile+'_FailReweight_weight_Down'].Write()
            hists[ifile].Write()
        else:
            hists[ifile].Scale(137/58.8)
            hists[ifile].Write()

def plot_datacards(VHHfiles,fpref, fplots_proce):
    addtional = fpref
    fsignal_procs = {"VHH_CV_0p5_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_0_kl_1_hbbhbb", "VHH_CV_1_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_1_kl_2_hbbhbb", "VHH_CV_1_C2V_2_kl_1_hbbhbb", "VHH_CV_1p5_C2V_1_kl_1_hbbhbb",'VHH_CV_1_C2V_1_kl_0_hbbhbb'}
    for ifile in VHHfiles:
        hsample = {}
        fin = R.TFile.Open('{0}'.format(ifile),'READ')
        #Get all plots
        MAX_ = 0
        MaxX_ = 0
        MinX_ = 0
        NX_ = 0
        for iprocs in fplots_proce:
            fin.cd()
            print("find ", iprocs, "in ", ifile )
            htemplte = fin.Get(iprocs).Clone()
            NX_=htemplte.GetNbinsX()
            MinX_=htemplte.GetBinCenter(1) - htemplte.GetBinWidth(1)/2.0
            MaxX_=htemplte.GetBinCenter(NX_) + htemplte.GetBinWidth(NX_)/2.0
            MAX_ = htemplte.GetMaximum() if htemplte.GetMaximum() > MAX_ else MAX_
            if iprocs in fsignal_procs:
                MAX_ = htemplte.GetMaximum()*100000 if htemplte.GetMaximum()*100000 > MAX_ else MAX_
            hsample[iprocs] = R.TH1F(iprocs,iprocs,NX_,1,1+NX_)
            hsample[iprocs].SetTitle(iprocs)
            hsample[iprocs].SetName(iprocs)
            for i in range(1,NX_+1):
                hsample[iprocs].SetBinContent(i,htemplte.GetBinContent(i))
                hsample[iprocs].SetBinError(i,htemplte.GetBinError(i))
        print(NX_,MinX_,MaxX_)
        #make plots
        c=R.TCanvas()
        c.SetFillColor(0)
        c.SetBorderMode(0)
        c.SetBorderSize(2)
        c.SetFrameBorderMode(0)
        frame_4fa51a0__1 = R.TH1D("frame_4fa51a0__1","",NX_,1,1+NX_)
        frame_4fa51a0__1.GetXaxis().SetTitle("transformed BDT variable")
        #frame_4fa51a0__1.GetXaxis().SetLabelFont(42)
        #frame_4fa51a0__1.GetXaxis().SetLabelSize(0.05)
        frame_4fa51a0__1.GetXaxis().SetTitleSize(0.05)
        frame_4fa51a0__1.GetXaxis().SetTitleOffset(0.8)
        frame_4fa51a0__1.GetXaxis().SetTitleFont(42)
        frame_4fa51a0__1.GetYaxis().SetTitle("")
        frame_4fa51a0__1.GetYaxis().SetLabelFont(42)
        frame_4fa51a0__1.GetYaxis().SetLabelSize(0.05)
        frame_4fa51a0__1.GetYaxis().SetTitleSize(0.05)
        frame_4fa51a0__1.GetYaxis().SetTitleFont(42)
        frame_4fa51a0__1.GetYaxis().SetRangeUser(0.1,MAX_*1.3)
        frame_4fa51a0__1.Draw("AXISSAME")
        R.gStyle.SetOptStat(0)
        c.SetLogy()
        leg = R.TLegend(0.15,0.2,0.55,0.45)
        leg.SetBorderSize(0)
        leg.SetLineStyle(1)
        leg.SetLineWidth(1)
        i=0
        for iprocs in fplots_proce:
            print(iprocs)
            if iprocs in fsignal_procs:
                hsample[iprocs].Scale(100000)
            hsample[iprocs].SetLineColor(1+i)
            #hsample[iprocs].SetLineColor(R.kRainBow+i*4)
            hsample[iprocs].Draw("same e")
            entry=leg.AddEntry(iprocs,iprocs)
            entry.SetFillStyle(1001)
            entry.SetMarkerStyle(8)
            entry.SetMarkerSize(1.5)
            entry.SetLineStyle(1)
            entry.SetLineWidth(3)
            entry.SetTextFont(42)
            entry.SetTextSize(0.06)
            i+=1
        leg.Draw()
        #c.SaveAs("%s_transform_%s.png"%(ifile[0:-5], addtional) )
        #c.SaveAs("%s_transform_%s.pdf"%(ifile[0:-5],addtional) )
        c.SaveAs("%s_transform_%s.C"%(ifile[0:-5],addtional) )

def ratioplot1(h1, h2, h1title, h2title):
   R.gStyle.SetOptStat(0)
   c1 = R.TCanvas("c1", "A ratio example")
   h1.GetXaxis().SetTitle(h1title)
   h1.GetYaxis().SetTitle(h2title)
   rp = R.TRatioPlot(h1, h2)
   c1.SetTicks(0, 1)
   rp.Draw()
   c1.Update()
   input()

def plot_BKGcomparison(VHHfiles, fplots_proce):
    addtional = 'compareBKG'
    for ifile in VHHfiles:
        hsample = {}
        fin = R.TFile.Open('{0}'.format(ifile),'READ')
        #Get all plots
        MAX_ = 0
        MaxX_ = 0
        MinX_ = 0
        NX_ = 0
        for iprocs in fplots_proce:
            fin.cd()
            print("find ", iprocs[0], "in ", ifile )
            htemplte = fin.Get(iprocs[0]).Clone()
            htemplte2 = fin.Get(iprocs[1]).Clone()
            NX_=htemplte.GetNbinsX()
            MinX_=htemplte.GetBinCenter(1) - htemplte.GetBinWidth(1)/2.0
            MaxX_=htemplte.GetBinCenter(NX_) + htemplte.GetBinWidth(NX_)/2.0
            MAX_ = htemplte.GetMaximum() if htemplte.GetMaximum() > MAX_ else MAX_
            hsample[iprocs[0]] = R.TH1F(iprocs[0],iprocs[0],NX_,1,1+NX_)
            hsample[iprocs[1]] = R.TH1F(iprocs[1],iprocs[1],NX_,1,1+NX_)
            hsample[iprocs[1]+'_ratio']= R.TH1F(iprocs[1]+'_ratio',iprocs[1]+'_ratio',NX_,1,1+NX_)
            for i in range(1,NX_+1):
                hsample[iprocs[0]].SetBinContent(i,htemplte.GetBinContent(i))
                hsample[iprocs[0]].SetBinError(i,htemplte.GetBinError(i))
                hsample[iprocs[1]].SetBinContent(i,htemplte2.GetBinContent(i))
                hsample[iprocs[1]].SetBinError(i,htemplte2.GetBinError(i))
                if htemplte2.GetBinContent(i)!=0:
                    hsample[iprocs[1]+'_ratio'].SetBinContent(i,htemplte.GetBinContent(i)/htemplte2.GetBinContent(i))
                    hsample[iprocs[1]+'_ratio'].SetBinError(i,htemplte.GetBinError(i)/htemplte2.GetBinContent(i))
                else:
                    hsample[iprocs[1]+'_ratio'].SetBinContent(i,0)
                    hsample[iprocs[1]+'_ratio'].SetBinError(i,0)

        basecan=R.TCanvas("basecan", "A ratio example",10,32,700,500)
        R.gStyle.SetOptStat(0)
        basecan.SetHighLightColor(2)
        basecan.Range(0,0,1,1)
        basecan.SetFillColor(0)
        basecan.SetBorderMode(0)
        basecan.SetBorderSize(2)
        basecan.SetTicky(1)
        basecan.SetFrameBorderMode(0)

        #Top pads
        pad = R.TPad("pad", "",0.0025,0.3,0.9975,0.9975)
        pad.Draw()
        pad.cd()
        pad.SetFillColor(0)
        pad.SetBorderMode(0)
        pad.SetBorderSize(2)
        pad.SetLogy()
        pad.SetBottomMargin(0.05)
        pad.SetFrameBorderMode(0)
        pad.SetFrameBorderMode(0)
        pad.SetLogy()
        frame_4fa51a0__1 = R.TH1D("frame_4fa51a0__1","",NX_,1,1+NX_)
        frame_4fa51a0__1.GetXaxis().SetTitle("transformed BDT variable")
        frame_4fa51a0__1.GetXaxis().SetTitleSize(0.05)
        frame_4fa51a0__1.GetXaxis().SetTitleOffset(0.8)
        frame_4fa51a0__1.GetXaxis().SetTitleFont(42)
        frame_4fa51a0__1.GetYaxis().SetTitle("")
        frame_4fa51a0__1.GetYaxis().SetLabelFont(42)
        frame_4fa51a0__1.GetYaxis().SetLabelSize(0.05)
        frame_4fa51a0__1.GetYaxis().SetTitleSize(0.05)
        frame_4fa51a0__1.GetYaxis().SetTitleFont(42)
        frame_4fa51a0__1.GetYaxis().SetRangeUser(0.001,MAX_*10)
        frame_4fa51a0__1.Draw("AXISSAME")

        leg = R.TLegend(0.61,0.6,0.8,0.85)
        leg.SetBorderSize(0)
        leg.SetLineStyle(1)
        leg.SetLineWidth(1)
        i=0
        colorb = [R.kRed,R.kCyan,R.kOrange,R.kBlue,R.kGreen]
        tmp=[]
        for iprocs in fplots_proce:
            tmp.append(iprocs)
            print(iprocs)
            hsample[tmp[i][0]].SetLineColor(colorb[2*i])
            hsample[tmp[i][0]].SetMarkerColor(colorb[2*i])
            hsample[tmp[i][0]].Draw("same e ")
            hsample[tmp[i][1]].SetLineColor(colorb[2*i+1])
            hsample[tmp[i][1]].SetMarkerColor(colorb[2*i+1])
            hsample[tmp[i][1]].Draw("same ep")
            hsample[tmp[i][0]].SetName(tmp[i][0])
            hsample[tmp[i][1]].SetName(tmp[i][1])
            hsample[tmp[i][0]].SetTitle(tmp[i][0])
            hsample[tmp[i][1]].SetTitle(tmp[i][1])
            entry=leg.AddEntry(tmp[i][0],tmp[i][0])
            entry.SetFillStyle(1001)
            entry.SetMarkerStyle(9)
            entry.SetMarkerSize(1.5)
            entry.SetLineStyle(1)
            entry.SetLineWidth(3)
            entry.SetTextFont(42)
            entry.SetTextSize(0.06)
            entry=leg.AddEntry(tmp[i][1],tmp[i][1])
            entry.SetFillStyle(1001)
            entry.SetMarkerStyle(8)
            entry.SetMarkerSize(1.5)
            entry.SetLineStyle(1)
            entry.SetLineWidth(3)
            entry.SetTextFont(42)
            entry.SetTextSize(0.06)
            i+=1
        basecan.cd()
        ##bootom
        bpad = R.TPad("bpad", "",0.0025,0.0025,0.9975,0.33)
        bpad.Draw()
        bpad.cd()
        bpad.SetFillColor(0)
        bpad.SetBorderMode(0)
        bpad.SetBorderSize(2)
        bpad.SetTopMargin(0.05)
        bpad.SetBottomMargin(0.3)
        bpad.SetFrameBorderMode(0)
        bpad.SetFrameBorderMode(0)
        frame_4fa51a0__2 = R.TH1D("frame_4fa51a0__2","",NX_,1,1+NX_)
        frame_4fa51a0__2.GetXaxis().SetTitle("transformed BDT variable")
        frame_4fa51a0__2.GetYaxis().SetTitle("Pass/FailReweight")
        frame_4fa51a0__2.GetXaxis().SetTitleSize(0.15)
        frame_4fa51a0__2.GetYaxis().SetNdivisions(3)
        frame_4fa51a0__2.GetXaxis().SetTitleOffset(0.8)
        frame_4fa51a0__2.GetXaxis().SetTitleFont(42)
        frame_4fa51a0__2.GetXaxis().SetLabelSize(0.1)
        frame_4fa51a0__2.GetYaxis().SetLabelFont(42)
        frame_4fa51a0__2.GetYaxis().SetLabelSize(0.1)
        frame_4fa51a0__2.GetYaxis().SetTitleSize(0.10)
        frame_4fa51a0__2.GetYaxis().SetTitleOffset(0.14)
        frame_4fa51a0__2.GetYaxis().SetTitleFont(42)
        frame_4fa51a0__2.GetYaxis().SetRangeUser(0.0,2)
        frame_4fa51a0__2.Draw("AXISSAME")
        i=0
        for iprocs in fplots_proce:
        #    print(iprocs)
            hsample[iprocs[1]+'_ratio'].SetLineColor(colorb[2*i+1])
            hsample[iprocs[1]+'_ratio'].Draw("same e")
        #    entry=leg.AddEntry(iprocs[1]+'_ratio',iprocs[1]+'_ratio')
            i+=1
        #input()
        basecan.cd()
        pad.cd()
        leg.Draw()
        tex = R.TLatex(0.14,0.95,"CMS")
        tex.SetNDC()
        tex.SetTextAlign(13)
        tex.SetTextSize(0.064)
        tex.SetLineWidth(2)
        tex.Draw()
        tex1 = R.TLatex(0.22,0.95,"Simulation Work in Progress")
        tex1.SetNDC()
        tex1.SetTextAlign(13)
        tex1.SetTextFont(52)
        tex1.SetTextSize(0.05)
        tex1.SetLineWidth(2)
        tex1.Draw()
        basecan.SaveAs("%s_transformBDT_%s.C"%(ifile[0:-5],addtional) )
        basecan.SaveAs("%s_transformBDT_%s.png"%(ifile[0:-5],addtional) )


def plot_BKGcomparison_updown(VHHfiles, fplots_proce):
    addtional = 'compareBKG'
    for ifile in VHHfiles:
        hsample = {}
        fin = R.TFile.Open('{0}'.format(ifile),'READ')
        #Get all plots
        MAX_ = 0
        MaxX_ = 0
        MinX_ = 0
        NX_ = 0
        for iprocs in fplots_proce:
            fin.cd()
            print("find ", iprocs[0], "in ", ifile )
            htemplte = fin.Get(iprocs[0]).Clone()
            htemplte2 = fin.Get(iprocs[1]).Clone()
            htemplte3 = fin.Get(iprocs[2]).Clone()
            NX_=htemplte.GetNbinsX()
            MinX_=htemplte.GetBinCenter(1) - htemplte.GetBinWidth(1)/2.0
            MaxX_=htemplte.GetBinCenter(NX_) + htemplte.GetBinWidth(NX_)/2.0
            MAX_ = htemplte.GetMaximum() if htemplte.GetMaximum() > MAX_ else MAX_
            hsample[iprocs[0]] = R.TH1F(iprocs[0],iprocs[0],NX_,1,1+NX_)
            hsample[iprocs[1]] = R.TH1F(iprocs[1],iprocs[1],NX_,1,1+NX_)
            hsample[iprocs[1]+'_ratio']= R.TH1F(iprocs[1]+'_ratio',iprocs[1]+'_ratio',NX_,1,1+NX_)
            hsample[iprocs[2]] = R.TH1F(iprocs[2],iprocs[2],NX_,1,1+NX_)
            hsample[iprocs[2]+'_ratio']= R.TH1F(iprocs[2]+'_ratio',iprocs[2]+'_ratio',NX_,1,1+NX_)
            for i in range(1,NX_+1):
                hsample[iprocs[0]].SetBinContent(i,htemplte.GetBinContent(i))
                hsample[iprocs[0]].SetBinError(i,htemplte.GetBinError(i))
                hsample[iprocs[1]].SetBinContent(i,htemplte2.GetBinContent(i))
                hsample[iprocs[1]].SetBinError(i,htemplte2.GetBinError(i))
                hsample[iprocs[2]].SetBinContent(i,htemplte3.GetBinContent(i))
                hsample[iprocs[2]].SetBinError(i,htemplte3.GetBinError(i))
                if htemplte.GetBinContent(i)!=0:
                    hsample[iprocs[1]+'_ratio'].SetBinContent(i,htemplte2.GetBinContent(i)/htemplte.GetBinContent(i))
                    hsample[iprocs[1]+'_ratio'].SetBinError(i,htemplte2.GetBinError(i)/htemplte.GetBinContent(i))
                    hsample[iprocs[2]+'_ratio'].SetBinContent(i,htemplte3.GetBinContent(i)/htemplte.GetBinContent(i))
                    hsample[iprocs[2]+'_ratio'].SetBinError(i,htemplte3.GetBinError(i)/htemplte.GetBinContent(i))
                else:
                    hsample[iprocs[1]+'_ratio'].SetBinContent(i,0)
                    hsample[iprocs[1]+'_ratio'].SetBinError(i,0)
                    hsample[iprocs[2]+'_ratio'].SetBinContent(i,0)
                    hsample[iprocs[2]+'_ratio'].SetBinError(i,0)

        basecan=R.TCanvas("basecan", "A ratio example",10,32,700,500)
        R.gStyle.SetOptStat(0)
        basecan.SetHighLightColor(2)
        basecan.Range(0,0,1,1)
        basecan.SetFillColor(0)
        basecan.SetBorderMode(0)
        basecan.SetBorderSize(2)
        basecan.SetTicky(1)
        basecan.SetFrameBorderMode(0)

        #Top pads
        pad = R.TPad("pad", "",0.0025,0.3,0.9975,0.9975)
        pad.Draw()
        pad.cd()
        pad.SetFillColor(0)
        pad.SetBorderMode(0)
        pad.SetBorderSize(2)
        pad.SetLogy()
        pad.SetBottomMargin(0.05)
        pad.SetFrameBorderMode(0)
        pad.SetFrameBorderMode(0)
        pad.SetLogy()
        frame_4fa51a0__1 = R.TH1D("frame_4fa51a0__1","",NX_,1,1+NX_)
        frame_4fa51a0__1.GetXaxis().SetTitle("transformed BDT variable")
        frame_4fa51a0__1.GetXaxis().SetTitleSize(0.05)
        frame_4fa51a0__1.GetXaxis().SetTitleOffset(0.8)
        frame_4fa51a0__1.GetXaxis().SetTitleFont(42)
        frame_4fa51a0__1.GetYaxis().SetTitle("")
        frame_4fa51a0__1.GetYaxis().SetLabelFont(42)
        frame_4fa51a0__1.GetYaxis().SetLabelSize(0.05)
        frame_4fa51a0__1.GetYaxis().SetTitleSize(0.05)
        frame_4fa51a0__1.GetYaxis().SetTitleFont(42)
        frame_4fa51a0__1.GetYaxis().SetRangeUser(0.01,MAX_*1.3)
        frame_4fa51a0__1.Draw("AXISSAME")

        leg = R.TLegend(0.61,0.6,0.8,0.85)
        leg.SetBorderSize(0)
        leg.SetLineStyle(1)
        leg.SetLineWidth(1)
        i=0
        colorb = [R.kRed,R.kCyan,R.kOrange,R.kBlue,R.kGreen,R.kYellow]
        for iprocs in fplots_proce:
            print(iprocs)
            hsample[iprocs[0]].SetLineColor(colorb[3*i])
            hsample[iprocs[0]].Draw("same e ")
            hsample[iprocs[1]].SetLineColor(colorb[3*i+1])
            hsample[iprocs[1]].Draw("same e")
            hsample[iprocs[2]].SetLineColor(colorb[3*i+2])
            hsample[iprocs[2]].Draw("same e")
            hsample[iprocs[0]].SetName(iprocs[0])
            hsample[iprocs[1]].SetName(iprocs[1]+'_Reweightup')
            hsample[iprocs[2]].SetName(iprocs[2]+'_Reweightdown')
            entry=leg.AddEntry(iprocs[0],iprocs[0])
            entry.SetFillStyle(1001)
            entry.SetMarkerStyle(8)
            entry.SetMarkerSize(1.5)
            entry.SetLineStyle(1)
            entry.SetLineWidth(3)
            entry.SetTextFont(42)
            entry.SetTextSize(0.06)
            entry=leg.AddEntry(iprocs[1]+'_Reweightup',iprocs[1]+'_Reweightup')
            entry.SetFillStyle(1001)
            entry.SetMarkerStyle(8)
            entry.SetMarkerSize(1.5)
            entry.SetLineStyle(1)
            entry.SetLineWidth(3)
            entry.SetTextFont(42)
            entry.SetTextSize(0.06)
            entry=leg.AddEntry(iprocs[2]+'_Reweightdown',iprocs[2]+'_Reweightdown')
            entry.SetFillStyle(1001)
            entry.SetMarkerStyle(8)
            entry.SetMarkerSize(1.5)
            entry.SetLineStyle(1)
            entry.SetLineWidth(3)
            entry.SetTextFont(42)
            entry.SetTextSize(0.06)
            i+=1
        leg.Draw()
        basecan.cd() 
        ##bootom
        bpad = R.TPad("bpad", "",0.0025,0.0025,0.9975,0.302)
        bpad.Draw()
        bpad.cd()
        bpad.SetFillColor(0)
        bpad.SetBorderMode(0)
        bpad.SetBorderSize(2)
        bpad.SetTopMargin(0.05)
        bpad.SetBottomMargin(0.3)
        bpad.SetFrameBorderMode(0)
        bpad.SetFrameBorderMode(0)
        frame_4fa51a0__2 = R.TH1D("frame_4fa51a0__2","",NX_,1,1+NX_)
        frame_4fa51a0__2.GetXaxis().SetTitle("transformed BDT variable")
        frame_4fa51a0__2.GetYaxis().SetTitle("Raw/Reweight")
        frame_4fa51a0__2.GetXaxis().SetTitleSize(0.15)
        frame_4fa51a0__2.GetYaxis().SetNdivisions(3)
        frame_4fa51a0__2.GetXaxis().SetTitleOffset(0.8)
        frame_4fa51a0__2.GetXaxis().SetTitleFont(42)
        frame_4fa51a0__2.GetXaxis().SetLabelSize(0.1)
        frame_4fa51a0__2.GetYaxis().SetLabelFont(42)
        frame_4fa51a0__2.GetYaxis().SetLabelSize(0.1)
        frame_4fa51a0__2.GetYaxis().SetTitleSize(0.12)
        frame_4fa51a0__2.GetYaxis().SetTitleOffset(0.14)
        frame_4fa51a0__2.GetYaxis().SetTitleFont(42)
        frame_4fa51a0__2.GetYaxis().SetRangeUser(0.0,3)
        frame_4fa51a0__2.Draw("AXISSAME")
        i=0
        for iprocs in fplots_proce:
        #    print(iprocs)
            hsample[iprocs[1]+'_ratio'].SetLineColor(colorb[3*i+1])
            hsample[iprocs[1]+'_ratio'].Draw("same e")
            hsample[iprocs[2]+'_ratio'].SetLineColor(colorb[3*i+2])
            hsample[iprocs[2]+'_ratio'].Draw("same e")
        #    entry=leg.AddEntry(iprocs[1]+'_ratio',iprocs[1]+'_ratio')
            i+=1
        #input()
        basecan.SaveAs("%s_transform_%s.C"%(ifile[0:-5],addtional) )
        #basecan.SaveAs("%s_transform_%s.png"%(ifile[0:-5],addtional) )



# ------------------------- train BDT and make related plots -------------------------

def compare_train_test(clf, X_train, y_train, X_test, y_test, outputFilename, bins=30):
    '''compare train and test histogram'''

    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    plt.figure()
    fig, ax = plt.subplots(figsize=(6,5))
    plt.xlim([-1.05, 1.05])
    plt.ylim([0, 10])
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.savefig(OutDir+outputFilename+'bdt_dist.png')
    np.savez(OutDir+outputFilename+'bdt_dist.npz', center=center, decisions_0=decisions[0], decisions_1=decisions[1], decisions_2=decisions[2], decisions_3=decisions[3])

def correlations(data, outputFilename, extra_str, **kwds):
    """Calculate pairwise correlation between features.
    Extra arguments are passed on to DataFrame.corr()
    """
    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr(**kwds)
    plt.figure()
    fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))
    
    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)

    ax1.set_title("Correlations_"+extra_str)

    labels = corrmat.columns.values
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, minor=False)
        
    plt.tight_layout()
    plt.savefig(OutDir+outputFilename+extra_str+'correlation.png')
    
def roc_compare(ab,a, b, lab1, lab2):
    a=np.load(a)
    b=np.load(b)
    plt.figure()
    #plt.plot(aext['fpr'], aext['tpr'], lw=1, label='roc0', color='g')
    plt.plot(a['fpr'], a['tpr'], lw=1, label=lab1, color='b')
    plt.plot(b['fpr'], b['tpr'], lw=1, label=lab2, color='r')
    aext=np.load('plots_var_fix/Wlnu_Fix_normalmhtrain.npz')
    plt.plot(aext['fpr'], aext['tpr'], lw=1, label='mixed', color='c')

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Diag')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(ab+'roc_sum.png')

def svsb_compare(ab,a, b,lab1,lab2):
    a=np.load(a)
    b=np.load(b)
    plt.figure()
    an=a['svsb']**2
    bn=b['svsb']**2
    an[np.isinf(an)] = 0
    bn[np.isinf(bn)] = 0
    print(np.sqrt((an).sum()), np.sqrt((bn).sum()))
    aa = np.sqrt((an).sum())
    bb = np.sqrt((bn).sum())
    plt.plot(a['center'], a['svsb'], lw=1,color='b', label=lab1+' (sum %0.2f)'%(aa))
    plt.plot(b['center'], b['svsb'], lw=1,color='r', label=lab2+' (sum %0.2f)'%(bb))

    aext=np.load('plots_var_fix/Wlnu_Fix_normalmhbdt_flatten.npz')
    aaext = np.sqrt((aext['svsb']**2).sum())
    plt.plot(aext['center'], aext['svsb'], lw=1, label='mixed (sum %0.2f)'%(aaext), color='c')
    print(len(a['svsb']),len(b['svsb']),len(aext['svsb']))

    plt.xlabel('BDT output')
    plt.ylabel('S/sqrt(B)')
    plt.title('Sensitivity')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(ab+"svsb_sum.png")

def train_bdt(X, y, X_train, X_test, y_train, y_test, outputFilename, bdt, branch_nickname, savebdt=True):
    '''train bdt'''
    print("fit BDT ")
    bdt.fit(X_train, y_train)
    if savebdt:
        pl.dump(bdt, open(OutDir+outputFilename+"_BDTAdaBoost.pk","wb"))
        var_list = [
                    #('H1_pnet', 'F'),
                    #('H2_pnet', 'F'),
                    ('v_pt', 'F'),
                    ( 'VHH_H1_pT', 'F'),
                    ( 'VHH_H2_pT', 'F'),
                    ( 'VHH_H1_m', 'F'),
                    ( 'VHH_H2_m', 'F'),
                    ( 'VHH_HH_m', 'F'),
                    ( 'VHH_HH_pT', 'F'),
                    ( 'v_phi', 'F'),
                    ( 'VHH_H1_phi', 'F'),
                    ( 'VHH_H2_phi', 'F'),
                   ]
        tmva_outfile_xml = OutDir+outputFilename+'_out.xml'
        print("convert_bdt_sklearn_tmva : ")
        convert_bdt_sklearn_tmva(bdt, var_list, tmva_outfile_xml)
        print("bdt model saved ! ")

    #perfomance check
    print("On testing sample:")
    y_predicted = bdt.predict(X_test)
    print(classification_report(y_test, y_predicted, target_names=["background", "signal"]))
    print("Area under ROC curve: %.4f"%(roc_auc_score(y_test, bdt.decision_function(X_test))))

    print("On training sample:")
    y_predicted = bdt.predict(X_train)
    print(classification_report(y_train, y_predicted, target_names=["background", "signal"]))
    print("Area under ROC curve: %.4f"%(roc_auc_score(y_train, bdt.decision_function(X_train))))

    #ROC curve
    print("ROC curve on testing sample")
    from sklearn.metrics import roc_curve, auc

    decisions = bdt.decision_function(X_test)
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(y_test, decisions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Diag')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(OutDir+outputFilename+'roc.png')

    #BDT learning curve
    train_predictions = bdt.staged_predict(X_train)
    test_predictions = bdt.staged_predict(X_test)
    scores_test = []
    scores_train = []
    for i in test_predictions:
        scores_test.append(accuracy_score(i,y_test))
    for i in train_predictions:
        scores_train.append(accuracy_score(i,y_train))
    plt.figure(1)
    fig1, ax1 = plt.subplots()
    plt.title("learning curves")
    plt.xlabel("Training iterations")
    plt.ylabel("accuracy")
    plt.grid()
    plt.plot(range(0,len(scores_train),1), scores_train, 'o-', color="r",label="Training accuracy")
    plt.plot(range(0,len(scores_test),1), scores_test, 'o-', color="g",label="Testing accuracy")
    plt.legend(loc="best")
    plt.savefig(OutDir+outputFilename+'learning_curve.png')
    np.savez(OutDir+outputFilename+'train.npz', fpr=fpr, tpr=tpr, scores_train=scores_train, scores_test=scores_test )

    #Save output
    #y_predicted = bdt.decision_function(X)
    #y_predicted.dtype = [('bdt_score', np.float64)]
    #array2root(y_predicted, OutDir+outputFilename+"test-prediction.root", "BDToutput")

    #flatten_bdt_output(bdt, X_train, y_train, X_test, y_test, outputFilename)
    compare_train_test(bdt, X_train, y_train, X_test, y_test, outputFilename)

    #correlation plot
    import pandas.core.common as com
    from pandas.core.index import Index
    from pandas.plotting import scatter_matrix
    # Create a pandas DataFrame for our data
    # this provides many convenience functions
    # for exploring your dataset
    # need to reshape y so it is a 2D array with one column
    df = pd.DataFrame(np.hstack((X, y.reshape(y.shape[0], -1))),
                      columns=branch_nickname+['y'])
    bg = df.y < 0.5
    sig = df.y > 0.5
    # remove the y column from the correlation matrix
    # after using it to select background and signal
    correlations(df[bg].drop('y', 1), outputFilename, 'bkg')
    correlations(df[sig].drop('y', 1), outputFilename, 'sig')

def train_bdt_withweight(X, y, X_train, X_test, y_train, y_test,z_train, z_test, outputFilename, bdt, branch_nickname, savebdt=True):
    '''train bdt'''
    print("fit BDT ")
    bdt.fit(X_train, y_train, z_train)
    if savebdt:
        pl.dump(bdt, open(OutDir+outputFilename+"_BDTAdaBoost.pk","wb"))
        var_list = [
                    ('v_pt', 'F'),
                    ( 'VHH_H1_pT', 'F'),
                    ( 'VHH_H2_pT', 'F'),
                    ( 'VHH_H1_m', 'F'),
                    ( 'VHH_H2_m', 'F'),
                    ( 'VHH_HH_m', 'F'),
                    ( 'VHH_HH_pT', 'F'),
                    ( 'v_phi', 'F'),
                    ( 'VHH_H1_phi', 'F'),
                    ( 'VHH_H2_phi', 'F'),
                   ]
        tmva_outfile_xml = OutDir+outputFilename+'_out.xml'
        print("convert_bdt_sklearn_tmva : ")
        convert_bdt_sklearn_tmva(bdt, var_list, tmva_outfile_xml)
        print("bdt model saved ! ")

    #perfomance check
    print("On testing sample:")
    y_predicted = bdt.predict(X_test)
    print(classification_report(y_test, y_predicted, target_names=["background", "signal"]))
    print("Area under ROC curve: %.4f"%(roc_auc_score(y_test, bdt.decision_function(X_test))))

    print("On training sample:")
    y_predicted = bdt.predict(X_train)
    print(classification_report(y_train, y_predicted, target_names=["background", "signal"]))
    print("Area under ROC curve: %.4f"%(roc_auc_score(y_train, bdt.decision_function(X_train))))

    #ROC curve
    print("ROC curve on testing sample")
    from sklearn.metrics import roc_curve, auc

    decisions = bdt.decision_function(X_test)
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(y_test, decisions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Diag')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(OutDir+outputFilename+'roc.png')
    compare_train_test(bdt, X_train, y_train, X_test, y_test, outputFilename)

    #correlation plot
    import pandas.core.common as com
    from pandas.core.index import Index
    from pandas.plotting import scatter_matrix
    # Create a pandas DataFrame for our data
    # this provides many convenience functions
    # for exploring your dataset
    # need to reshape y so it is a 2D array with one column
    df = pd.DataFrame(np.hstack((X, y.reshape(y.shape[0], -1))),
                      columns=branch_nickname+['y'])
    bg = df.y < 0.5
    sig = df.y > 0.5
    # remove the y column from the correlation matrix
    # after using it to select background and signal
    correlations(df[bg].drop('y', 1), outputFilename, 'bkg')
    correlations(df[sig].drop('y', 1), outputFilename, 'sig')



# -------------------------------------------------------------
# ----------------  main training start    --------------------
# -------------------------------------------------------------

inputsigfiles = 'CV*C2V*AllCh_Boosted.root'
inputbkgfiles = 'Boosted/bkg_*_skim.root'
OutDir = 'test_1024_signalVsfailreweightbkg_use0.8asfail/'
createDir(OutDir)

# Read data
branch_names = [
                'selLeptons_pt_0',
                'VHHFatJet1_Pt',
                'VHHFatJet2_Pt',
                'VHHFatJet1_Msoftdrop',
                'VHHFatJet2_Msoftdrop',
                'VHHFatJet_mjj',
                'VHHFatJet_HHPt',
                'selLeptons_phi_0',
                'VHHFatJet1_phi',
                'VHHFatJet2_phi',
                ]
branch_nickname = [ #'H1_pnet','H2_pnet', 
              'lepton_pt','H1_pt','H2_pt','mh1','mh2','HHm','HHpT','lepton_phi','H1_phi','H2_phi']
passcut = 'weight<1&& (isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9'
passcut1 = 'weight<1&& (isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94'
passcut2 = 'weight<1&& (isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9&&!(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)'
failcut='weight<1&& (isWenu||isWmunu) && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)'
failcut='weight<1&& (isWenu||isWmunu) && (VHHFatJet1_ParticleNetMD_bbvsQCD>0.8 || VHHFatJet2_ParticleNetMD_bbvsQCD>0.8) && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)'

# -------------------------------------------------------------
# -----------------   train_reweight  -------------------------
# -------------------------------------------------------------

train_rew_bdt_single=False
if train_rew_bdt_single:
    branch_names.append("weight")
    #Bkg pass
    bkgpass = root2array(inputbkgfiles, "Events", branch_names,selection=passcut)
    bkgpass = rec2array(bkgpass)
    bkgfail = root2array(inputbkgfiles, "Events", branch_names,selection=failcut)
    bkgfail = rec2array(bkgfail)
    #bkgfail = bkgfail[200000:300000,:]
    dt_rew_0p9 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05, random_state=True)
    bdt_rew_0p9 = AdaBoostClassifier(dt_rew_0p9,
                             algorithm='SAMME',
                             n_estimators=200,
                             learning_rate=0.5)
    X_rew = np.concatenate((bkgpass, bkgfail))
    y_rew = np.concatenate((np.ones(bkgpass.shape[0]), np.zeros(bkgfail.shape[0])))
    X_train_rew, X_test_rew, y_train_rew,y_test_rew = train_test_split(X_rew, y_rew, test_size=0.2, random_state=327)
    outputFilename = 'AllB_TTAll_kine_rew_0p9_'
    # do trainning
    #train_bdt(X_rew[:,0:-1], y_rew, X_train_rew[:,0:-1], X_test_rew[:,0:-1], y_train_rew, y_test_rew, outputFilename, bdt_rew_0p9, branch_nickname, savebdt=True)
    #exit()

train_rew_bdt=False
if train_rew_bdt:
    branch_names.append("weight")
    #Bkg pass
    bkgpass1 = root2array(inputbkgfiles, "Events", branch_names,selection=passcut1)
    bkgpass1 = rec2array(bkgpass1)
    print("bkgpass len: ", len(bkgpass1))
    print(bkgpass1[0])
    bkgpass2 = root2array(inputbkgfiles, "Events", branch_names,selection=passcut2)
    bkgpass2 = rec2array(bkgpass2)
    print("bkgpass len: ", len(bkgpass2))
    print(bkgpass2[0])
    #Bkg fail
    bkgfail = root2array(inputbkgfiles, "Events", branch_names,selection=failcut)
    bkgfail = rec2array(bkgfail)
    bkgfail1, bkgfail2 = train_test_split(bkgfail, test_size=0.5, random_state=327)
    #bkgfail1 = bkgfail[0:100000,:]
    #bkgfail2 = bkgfail[100000:200000,:]
    print("bkgfail len: ", len(bkgfail))
    print(bkgfail[0])
    ## ReWeight
    dt_rew_0p9_0p94 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05, random_state=True)
    bdt_rew_0p9_0p94 = AdaBoostClassifier(dt_rew_0p9_0p94,
                             algorithm='SAMME',
                             n_estimators=200,
                             learning_rate=0.5)
    dt_rew_0p94 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05, random_state=True)
    bdt_rew_0p94 = AdaBoostClassifier(dt_rew_0p94,
                             algorithm='SAMME',
                             n_estimators=200,
                             learning_rate=0.5)
    X_rew1 = np.concatenate((bkgpass1, bkgfail1))
    y_rew1 = np.concatenate((np.ones(bkgpass1.shape[0]), np.zeros(bkgfail1.shape[0])))
    X_train_rew1, X_test_rew1, y_train_rew1,y_test_rew1 = train_test_split(X_rew1, y_rew1, test_size=0.2, random_state=327)
    outputFilename = 'AllB_TTAll_kine_rew_0p94_'
    # do trainning
    #train_bdt(X_rew1[:,0:-1], y_rew1, X_train_rew1[:,0:-1], X_test_rew1[:,0:-1], y_train_rew1, y_test_rew1, outputFilename, bdt_rew_0p94, branch_nickname, savebdt=True)
    print("len X_test_rew:", np.shape(X_test_rew1))
    print("len y_test_rew:",np.shape(y_test_rew1))   
    X_rew2 = np.concatenate((bkgpass2, bkgfail2))
    y_rew2 = np.concatenate((np.ones(bkgpass2.shape[0]), np.zeros(bkgfail2.shape[0])))
    X_train_rew2, X_test_rew2, y_train_rew2,y_test_rew2 = train_test_split(X_rew2, y_rew2, test_size=0.2, random_state=327)
    outputFilename = 'AllB_TTAll_kine_rew_0p9_0p94_'
    # do trainning
    #train_bdt(X_rew2[:,0:-1], y_rew2, X_train_rew2[:,0:-1], X_test_rew2[:,0:-1], y_train_rew2, y_test_rew2, outputFilename, bdt_rew_0p9_0p94, branch_nickname, savebdt=True)
    print("len X_test_rew:", np.shape(X_test_rew2))
    print("len y_test_rew:",np.shape(y_test_rew2))
    #exit()

inputpicklefile='reweightBDT/AllB_TTAll_kine_rew_0p9__BDTAdaBoost.pk'
bdt_rew = pl.load(open(inputpicklefile,"rb"))
#print("bdt.feature_importances_ :",bdt_rew.feature_importances_)

inputpicklefile='reweightBDT/AllB_TTAll_kine_rew_0p94__BDTAdaBoost.pk'
bdt_rew1 = pl.load(open(inputpicklefile,"rb"))
#print("bdt.feature_importances_ :",bdt_rew1.feature_importances_)

inputpicklefile='reweightBDT/AllB_TTAll_kine_rew_0p9_0p94__BDTAdaBoost.pk'
bdt_rew2 = pl.load(open(inputpicklefile,"rb"))
#print("bdt.feature_importances_ :",bdt_rew2.feature_importances_)


# -----------------------------------------------------------------------
# ------------------------- train_SvB with pass bkg ---------------------
# -----------------------------------------------------------------------

train_SvB_bdt=False
if train_SvB_bdt:
    inputsigfiles = 'Boosted/*CV_*_0_skim.root'
    inputbkgfiles = 'Boosted/bkg_*_skim.root'
    OutDir = 'test_1026_B_svb_norew/'
    createDir(OutDir)
    branch_names = [
                    'h1_pnetcat','h2_pnetcat',
                    'selLeptons_pt_0',
                    'VHHFatJet1_Pt',
                    'VHHFatJet2_Pt',
                    'VHHFatJet1_Msoftdrop',
                    'VHHFatJet2_Msoftdrop',
                    'VHHFatJet_mjj',
                    'VHHFatJet_HHPt',
                    'selLeptons_phi_0',
                    'VHHFatJet1_phi',
                    'VHHFatJet2_phi',
                    ]
    branch_nickname = [ 'h1_pnetcat','h2_pnetcat','lepton_pt','H1_pt','H2_pt','mh1','mh2','HHm','HHpT','lepton_phi','H1_phi','H2_phi']
    passcut = 'weight<1&& (isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9'

    #Bkg pass
    bkgpass = root2array(inputbkgfiles, "Events", branch_names,selection=passcut)
    bkgpass = rec2array(bkgpass)
    print("bkgpass len: ", len(bkgpass))
    print(bkgpass[0])   
    #Signal
    signal = root2array(inputsigfiles, "Events", branch_names,selection=passcut)
    signal = rec2array(signal)
    print("sig len: ", len(signal))
    print(signal[0])
    ## SvB
    dt_svb = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.05, random_state=True)
    bdt_svb = AdaBoostClassifier(dt_svb,
                             algorithm='SAMME',
                             n_estimators=800,
                             learning_rate=0.5)
    
    X_svb= np.concatenate((signal, bkgpass))
    y_svb = np.concatenate((np.ones(signal.shape[0]), np.zeros(bkgpass.shape[0])))
    X_train_svb, X_test_svb, y_train_svb, y_test_svb = train_test_split(X_svb, y_svb, test_size=0.33, random_state=492)
    
    outputFilename = 'WlnB_svb_norew'
    #train_bdt(X_svb, y_svb, X_train_svb, X_test_svb, y_train_svb, y_test_svb, outputFilename, bdt_svb,branch_nickname, savebdt=True)

train_SvB_bdt=True
if train_SvB_bdt:
    # make datacards
    branch_names = [
                    'h1_pnetcat','h2_pnetcat',
                    'selLeptons_pt_0',
                    'VHHFatJet1_Pt',
                    'VHHFatJet2_Pt',
                    'VHHFatJet1_Msoftdrop',
                    'VHHFatJet2_Msoftdrop',
                    'VHHFatJet_mjj',
                    'VHHFatJet_HHPt',
                    'selLeptons_phi_0',
                    'VHHFatJet1_phi',
                    'VHHFatJet2_phi',
                    ]
    inputpicklefile='test_1026_B_svb_norew/WlnB_svb_norew_BDTAdaBoost.pk'
    bdt_norew = pl.load(open(inputpicklefile,"rb"))    
    OutDir = 'test_1026_B_datacards_norew/'
    createDir(OutDir)
    procs_rew = ["VHH_CV_0p5_C2V_1_kl_1_hbbhbb",
                 "VHH_CV_1_C2V_0_kl_1_hbbhbb", "VHH_CV_1_C2V_1_kl_1_hbbhbb",
                 "VHH_CV_1_C2V_1_kl_2_hbbhbb", "VHH_CV_1_C2V_2_kl_1_hbbhbb",
                 "VHH_CV_1p5_C2V_1_kl_1_hbbhbb","TT","TTB","data_obs",
                 'VHH_CV_1_C2V_1_kl_0_hbbhbb','VHH_CV_1_C2V_1_kl_20_hbbhbb']
    
    signal_procs = ["VHH_CV_0p5_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_0_kl_1_hbbhbb",
                    "VHH_CV_1_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_1_kl_2_hbbhbb",
                    "VHH_CV_1_C2V_2_kl_1_hbbhbb", "VHH_CV_1p5_C2V_1_kl_1_hbbhbb",
                    'VHH_CV_1_C2V_1_kl_0_hbbhbb','VHH_CV_1_C2V_1_kl_20_hbbhbb']
    procs = ["VHH_CV_0p5_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_0_kl_1_hbbhbb",
             "VHH_CV_1_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_1_kl_2_hbbhbb",
             "VHH_CV_1_C2V_2_kl_1_hbbhbb", "VHH_CV_1p5_C2V_1_kl_1_hbbhbb",
             "TT","TTB","data_obs",'VHH_CV_1_C2V_1_kl_0_hbbhbb','VHH_CV_1_C2V_1_kl_20_hbbhbb']
    filename = {
            'TT': 'Boosted/bkg_TT_skim.root',
            'TTB': 'Boosted/bkg_ttbb_skim.root',
            'VHH_CV_1_C2V_1_kl_1_hbbhbb': 'Boosted/NLO_CV_1_0_C2V_1_0_C3_1_0_skim.root',
            'VHH_CV_0p5_C2V_1_kl_1_hbbhbb': 'Boosted/CV_0_5_C2V_1_0_C3_1_0_skim.root',
            'VHH_CV_1_C2V_0_kl_1_hbbhbb': 'Boosted/CV_1_0_C2V_0_0_C3_1_0_skim.root',
            'VHH_CV_1_C2V_1_kl_2_hbbhbb': 'Boosted/CV_1_0_C2V_1_0_C3_2_0_skim.root',
            'VHH_CV_1_C2V_2_kl_1_hbbhbb': 'Boosted/CV_1_0_C2V_2_0_C3_1_0_skim.root',
            'VHH_CV_1p5_C2V_1_kl_1_hbbhbb': 'Boosted/CV_1_5_C2V_1_0_C3_1_0_skim.root',
            'VHH_CV_1_C2V_1_kl_0_hbbhbb': 'Boosted/CV_1_0_C2V_1_0_C3_0_0_skim.root',
            'VHH_CV_1_C2V_1_kl_20_hbbhbb': 'Boosted/CV_1_0_C2V_1_0_C3_20_0_skim.root',
    }
    ttbbstitching = '&& ( (sampleIndex==202||sampleIndex==203||sampleIndex==212)*(IsttB==0) || (sampleIndex==24||sampleIndex==25||sampleIndex==26)*(IsttB>0) || (sampleIndex<0))'
    passcut = "isWenu && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9"+ttbbstitching
    flattencut = "(isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9"+ttbbstitching
    outputfile=OutDir+'datacards_Wenu_svb_norew_0p9_1.root'
    makedatacards_norew(outputfile, passcut , flattencut, procs_rew, filename,branch_names,bdt_norew)

    passcut = "isWmunu && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9"+ttbbstitching
    outputfile=OutDir+'datacards_Wmunu_svb_norew_0p9_1.root'
    makedatacards_norew(outputfile, passcut , flattencut,procs_rew, filename,branch_names,bdt_norew)
    plots_proce = [["TT","TTB"]]
    plot_BKGcomparison([outputfile], plots_proce)
 

exit()


# -------------------------------------------------------------
# ------------- train_SvB with fail reweight  -----------------
# -------------------------------------------------------------

import bisect
train_SvB_bdt_with_reweightedFailB=False
if train_SvB_bdt_with_reweightedFailB:
    #inputsigfiles = 'Boosted/CV_1_0_C2V_1_0_C3_2*_0_skim.root'
    inputsigfiles = 'Boosted/*CV_*_0_skim.root'
    #inputsigfiles = 'Boosted/CV_1_0_C2V_2_0_C3_1_0_skim.root'
    inputbkgfiles = 'Boosted/bkg_*_skim.root'
    OutDir = 'test_1025_B_svb/'
    createDir(OutDir)
    branch_names = [
                    'selLeptons_pt_0',
                    'VHHFatJet1_Pt',
                    'VHHFatJet2_Pt',
                    'VHHFatJet1_Msoftdrop',
                    'VHHFatJet2_Msoftdrop',
                    'VHHFatJet_mjj',
                    'VHHFatJet_HHPt',
                    'selLeptons_phi_0',
                    'VHHFatJet1_phi',
                    'VHHFatJet2_phi',
                    ]
    branch_nickname = [ 'lepton_pt','H1_pt','H2_pt','mh1','mh2','HHm','HHpT','lepton_phi','H1_phi','H2_phi']
    passcut = 'weight<1&& (isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9'
    passcut1 = 'weight<1&& (isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94'
    passcut2 = 'weight<1&& (isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9&&!(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)'
    failcut='weight<1&& (isWenu||isWmunu) && (VHHFatJet1_ParticleNetMD_bbvsQCD>0.8 || VHHFatJet2_ParticleNetMD_bbvsQCD>0.8) && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)'

    #Load reweight BDT
    inputpicklefile='reweightBDT/AllB_TTAll_kine_rew_0p9__BDTAdaBoost.pk'
    #bdt_rew = pl.load(open(inputpicklefile,"rb"))
    inputpicklefile='reweightBDT/AllB_TTAll_kine_rew_0p94__BDTAdaBoost.pk'
    #bdt_rew = pl.load(open(inputpicklefile,"rb"))
    inputpicklefile='reweightBDT/AllB_TTAll_kine_rew_0p9_0p94__BDTAdaBoost.pk'
    bdt_rew = pl.load(open(inputpicklefile,"rb"))


    bkgpass = root2array(inputbkgfiles, "Events", branch_names,selection=passcut)
    bkgpass = rec2array(bkgpass)
    bkgfail = root2array(inputbkgfiles, "Events", branch_names,selection=failcut)
    bkgfail = rec2array(bkgfail)
    #bkgfail  = train_test_split(bkgfail, test_size=0.5, random_state=327)
    #bkgfail = bkgfail[0:200000] #0p9_0p94
    #bkgfail = bkgfail[200000:400000] #0p94
    bkgpass_rewBDT = bdt_rew.decision_function(pd.DataFrame(bkgpass))
    bkgfail_rewBDT = bdt_rew.decision_function(pd.DataFrame(bkgfail))

    print("============ Flatten passed reweight_BDT  ===========")
    bbins_passrewBDTflatten = []
    if len(bbins_passrewBDTflatten) == 0:
        print("============ find best binning ===========")
        decisions = bkgpass_rewBDT
        low = min(np.min(d) for d in decisions)
        high = max(np.max(d) for d in decisions)
        low_high = (low,high)
        print("low_high",low_high)
        low=-1
        high=1
        # --------------------------------------   get flat binning
        s_tot = np.ones(len(decisions)).sum()
        bins = 30
        values = np.sort(decisions)
        cumu = np.cumsum( np.ones(len(decisions))  )
        targets = [n*s_tot/bins for n in range(1,bins)]
        workingPoints = []
        for target in targets:
            index = np.argmax(cumu>target)
            workingPoints.append(values[index])
        bbins_passrewBDTflatten = [float(low)]+workingPoints+[float(high)]
    #print(bbins_passrewBDTflatten)
    #print(len(bkgfail_rewBDT))
    #totl=0
    rew_weight = []
    for nele in range(len(bbins_passrewBDTflatten)-1):
        a=len(bkgfail_rewBDT[ (bkgfail_rewBDT>=bbins_passrewBDTflatten[nele]) & (bkgfail_rewBDT<bbins_passrewBDTflatten[nele+1]) ])
        #totl+=a
        #print('from ', bbins_passrewBDTflatten[nele],' to ',bbins_passrewBDTflatten[nele+1])#, ' is ',a,totl)
        b=len(bkgpass_rewBDT[ (bkgpass_rewBDT>=bbins_passrewBDTflatten[nele]) & (bkgpass_rewBDT<bbins_passrewBDTflatten[nele+1]) ])
        #print(a, b, b/a)
        rew_weight.append(b/a)
    #print(rew_weight)
    if len(bbins_passrewBDTflatten) != len(rew_weight)+1:
        print('we got problems bbins_passrewBDTflatten has len ',len(bbins_passrewBDTflatten), ' but rew_weight has len ',len(rew_weight))
        exit()
    # -----------------  get the weight for each failed bkg
    #print(bisect.bisect_left(bbins_passrewBDTflatten, 1))
    #bkgfail_weight=bkgfail_rewBDT[0:10]
    #print(len(bkgfail_rewBDT))
    #bkgfail_weight=[int(bisect.bisect_left(bbins_passrewBDTflatten, x)) for x in bkgfail_rewBDT]
    bkgfail_weight=[rew_weight[int(bisect.bisect_left(bbins_passrewBDTflatten, x)-1)] for x in bkgfail_rewBDT]
    #print(len(bkgfail_weight), bkgfail_weight)
    #exit()
    print("============ start training  ===========")
    #Signal
    signal = root2array(inputsigfiles, "Events", branch_names,selection=passcut)
    signal = rec2array(signal)
    print("sig len: ", len(signal))
    print(signal[0])
    ## SvB
    dt_svb = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.05, random_state=True)
    bdt_svb = AdaBoostClassifier(dt_svb,
                             algorithm='SAMME',
                             n_estimators=500,
                             learning_rate=0.5)

    X_svb= np.concatenate((signal, bkgfail))
    y_svb = np.concatenate((np.ones(signal.shape[0]), np.zeros(bkgfail.shape[0])))
    z_svb= np.concatenate((np.ones(signal.shape[0]), bkgfail_weight))
    X_train_svb, X_test_svb, y_train_svb, y_test_svb, z_train_svb, z_test_svb = train_test_split(X_svb, y_svb,z_svb, test_size=0.33, random_state=492)
    #outputFilename = 'WlnB_svb_gt_0p9_lt_0p94_'
    #outputFilename = 'WlnB_signalc3_20or2_svb_0p9_1'
    outputFilename = 'WlnB_signalall_svb_0p9_0p94'
    #outputFilename = 'WlnB_signalc2v_2_svb_0p94_1'
    #outputFilename = 'WlnB_svb_gt_0p9_'
    #print(X_svb[0,0:-1])
    train_bdt_withweight(X_svb, y_svb, X_train_svb, X_test_svb, y_train_svb, y_test_svb,z_train_svb, z_test_svb, outputFilename, bdt_svb,branch_nickname, savebdt=True)
    exit()    


#Load BDT
#test_1025_B_svb/WlnB_signalc2v_2_svb_0p94_1_BDTAdaBoost.pk    test_1025_B_svb/WlnB_signalc3_20or2_svb_0p9_1_BDTAdaBoost.pk

inputpicklefile='test_1025_B_svb/WlnB_signalall_svb_0p9_1_BDTAdaBoost.pk'
bdt_svb = pl.load(open(inputpicklefile,"rb"))
inputpicklefile='test_1025_B_svb/WlnB_signalall_svb_0p94_1_BDTAdaBoost.pk'
bdt_svb1 = pl.load(open(inputpicklefile,"rb"))
#print(bdt_svb1.feature_importances_)
inputpicklefile='test_1025_B_svb/WlnB_signalall_svb_0p9_0p94_BDTAdaBoost.pk'
bdt_svb2 = pl.load(open(inputpicklefile,"rb"))
#print(bdt_svb2.feature_importances_)


#
#
## -------------------------------------------------------------
## study the feature in BDT
## -------------------------------------------------------------
#
##passcut = ' (isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94 '
#passcut = ' (isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9 && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)'
#
##Bkg pass
#bkgpass = root2array(inputbkgfiles, "Events", branch_names,selection=passcut)
#bkgpass = rec2array(bkgpass)
#print("bkgpass len: ", len(bkgpass))
#print(bkgpass[0,:])
##Signal
#signal = root2array(inputsigfiles, "Events", branch_names,selection=passcut)
#signal = rec2array(signal)
#print("sig len: ", len(signal))
#print(signal[0,:])
#
##0.94
##BDTDecision1 = bdt_svb1.decision_function(pd.DataFrame(signal[:,:]))
##BDTDecision2 = bdt_svb1.decision_function(pd.DataFrame(bkgpass[:,:]))
##0.9-0.94
#BDTDecision1 = bdt_svb2.decision_function(pd.DataFrame(signal[:,:]))
#BDTDecision2 = bdt_svb2.decision_function(pd.DataFrame(bkgpass[:,:]))
#
#makeplt=True
#
#if makeplt:
#    #1D
#    low1 = min(np.min(d) for d in BDTDecision1)
#    high1 = max(np.max(d) for d in BDTDecision1)
#    low2 = min(np.min(d) for d in BDTDecision2)
#    high2 = max(np.max(d) for d in BDTDecision2)
#    low_high = (min(low1,low2),max(high1,high2))
#    plt.figure()
#    fig, ax = plt.subplots(figsize=(6,5))
#    plt.xlim([-1.05, 1.05])
#    plt.ylim([0, 10])
#    bins=30
#    plt.hist(BDTDecision1,
#             color='r', alpha=0.5, range=low_high, bins=bins,
#             histtype='stepfilled', density=True,
#             label='S (passed)')
#    plt.hist(BDTDecision2,
#             color='b', alpha=0.5, range=low_high, bins=bins,
#             histtype='stepfilled', density=True,
#             label='B (passed)')
#
#    plt.xlabel("BDT output")
#    plt.ylabel("Arbitrary units")
#    plt.legend(loc='best')
#    #plt.savefig(OutDir+outputFilename+'bdt_dist.png')
#    plt.show()
#    input('finished')
#
#    #2D 
#    # Big bins
#    #plt.hist2d(BDTDecision2, bkgpass[:,0], bins=(200, 800), cmap=plt.cm.Reds)
#    #plt.ylim([0, 200])
#    #plt.xlabel("BDT output")
#    #plt.ylabel("Lepton pt (GeV)")
#    #plt.show()
#    #plt.savefig('SvBBDT_lep_bdt_dist.png')
#
#
#exit()
#
# -------------------------------------------------------------

print("------------------ BDT training finished  -------------------")
# -------------------------------------------------------------
# ------------------ BDT training finished  -------------------
# -------------------------------------------------------------




# -------------------------------------------------------------
# ------------------ Start reweight validation -------------------
# -------------------------------------------------------------

print("------------------ Start reweight validation  -------------------")

#evaluate train
#BDTDecision1 = bdt_rew1.decision_function(pd.DataFrame(X_train_rew1[:,0:-1]))
#outputFilename = 'extractWeight_and_Do_validation_train__0p94.root'
#extract_reWeight_Dovalidation(outputFilename, X_train_rew1, BDTDecision1, y_train_rew1)
#plot_BDTreweight([OutDir+outputFilename], ['histspass','histsfail','histsweight'],'test')
#evaluate all

#BDTDecision = bdt_rew.decision_function(pd.DataFrame(X_rew[:,0:-1]))
#outputFilename = 'extractWeight_validation_TT_0p9_1.root'
#extract_reWeight_Dovalidation(outputFilename, X_rew, BDTDecision, y_rew)
#plot_BDTreweight([OutDir+outputFilename], ['histspass','histsfail','histsweight'],'all')
#plot_hist = ['VHHFatJet_HHPt','VHHFatJet_mjj','selLeptons_pt_0','VHHFatJet1_Pt','VHHFatJet2_Pt','VHHFatJet1_Msoftdrop','VHHFatJet2_Msoftdrop']#,'VHHFatJet_mjj','VHHFatJet_HHPt']#,'selLeptons_phi_0','VHHFatJet1_phi','VHHFatJet2_phi']
#plot_reweighted_inputs([OutDir+outputFilename], plot_hist,'TT')

#BDTDecision1 = bdt_rew1.decision_function(pd.DataFrame(X_rew1[:,0:-1]))
#outputFilename = 'extractWeight_validation_TT_0p94_1.root'
#extract_reWeight_Dovalidation(outputFilename, X_rew1, BDTDecision1, y_rew1)
#plot_BDTreweight([OutDir+outputFilename], ['histspass','histsfail','histsweight'],'all')
#plot_hist = ['VHHFatJet_HHPt','VHHFatJet_mjj','selLeptons_pt_0','VHHFatJet1_Pt','VHHFatJet2_Pt','VHHFatJet1_Msoftdrop','VHHFatJet2_Msoftdrop']#,'VHHFatJet_mjj','VHHFatJet_HHPt']#,'selLeptons_phi_0','VHHFatJet1_phi','VHHFatJet2_phi']
#plot_reweighted_inputs([OutDir+outputFilename], plot_hist,'TT')

#BDTDecision2 = bdt_rew2.decision_function(pd.DataFrame(X_rew2[:,0:-1]))
#outputFilename = 'extractWeight_validation_TT_0p9_0p94.root'
#extract_reWeight_Dovalidation(outputFilename, X_rew2, BDTDecision2, y_rew2)
#plot_BDTreweight([OutDir+outputFilename], ['histspass','histsfail','histsweight'],'all')
#plot_hist = ['VHHFatJet_HHPt','VHHFatJet_mjj','selLeptons_pt_0','VHHFatJet1_Pt','VHHFatJet2_Pt','VHHFatJet1_Msoftdrop','VHHFatJet2_Msoftdrop']#,'VHHFatJet_mjj','VHHFatJet_HHPt']#,'selLeptons_phi_0','VHHFatJet1_phi','VHHFatJet2_phi']
#plot_reweighted_inputs([OutDir+outputFilename], plot_hist,'TT')


#exit()

# -------------------------------------------------------------
# ------------------ Start making datacards  -------------------
# -------------------------------------------------------------

print("------------------ Start making datacards  -------------------")


OutDir = 'test_1025_B_datacards_all30bins/'
#OutDir = 'test_1025_B_datacards_c2v/'
#OutDir = 'test_1025_B_datacards_3/'
createDir(OutDir)

#Make datacards
procs_rew = ["VHH_CV_0p5_C2V_1_kl_1_hbbhbb", 
             "VHH_CV_1_C2V_0_kl_1_hbbhbb", "VHH_CV_1_C2V_1_kl_1_hbbhbb", 
             "VHH_CV_1_C2V_1_kl_2_hbbhbb", "VHH_CV_1_C2V_2_kl_1_hbbhbb", 
             "VHH_CV_1p5_C2V_1_kl_1_hbbhbb","TT","TTB","data_obs",
             'VHH_CV_1_C2V_1_kl_0_hbbhbb','VHH_CV_1_C2V_1_kl_20_hbbhbb']

signal_procs = ["VHH_CV_0p5_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_0_kl_1_hbbhbb", 
                "VHH_CV_1_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_1_kl_2_hbbhbb", 
                "VHH_CV_1_C2V_2_kl_1_hbbhbb", "VHH_CV_1p5_C2V_1_kl_1_hbbhbb",
                'VHH_CV_1_C2V_1_kl_0_hbbhbb','VHH_CV_1_C2V_1_kl_20_hbbhbb']
procs = ["VHH_CV_0p5_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_0_kl_1_hbbhbb", 
         "VHH_CV_1_C2V_1_kl_1_hbbhbb", "VHH_CV_1_C2V_1_kl_2_hbbhbb", 
         "VHH_CV_1_C2V_2_kl_1_hbbhbb", "VHH_CV_1p5_C2V_1_kl_1_hbbhbb",
         "TT","TTB","data_obs",'VHH_CV_1_C2V_1_kl_0_hbbhbb','VHH_CV_1_C2V_1_kl_20_hbbhbb']

filename = {
        'TT': 'Boosted/bkg_TT_skim.root',
        'TTB': 'Boosted/bkg_ttbb_skim.root',
        'VHH_CV_1_C2V_1_kl_1_hbbhbb': 'Boosted/NLO_CV_1_0_C2V_1_0_C3_1_0_skim.root', 
        'VHH_CV_0p5_C2V_1_kl_1_hbbhbb': 'Boosted/CV_0_5_C2V_1_0_C3_1_0_skim.root',
        'VHH_CV_1_C2V_0_kl_1_hbbhbb': 'Boosted/CV_1_0_C2V_0_0_C3_1_0_skim.root',
        'VHH_CV_1_C2V_1_kl_2_hbbhbb': 'Boosted/CV_1_0_C2V_1_0_C3_2_0_skim.root',
        'VHH_CV_1_C2V_2_kl_1_hbbhbb': 'Boosted/CV_1_0_C2V_2_0_C3_1_0_skim.root',
        'VHH_CV_1p5_C2V_1_kl_1_hbbhbb': 'Boosted/CV_1_5_C2V_1_0_C3_1_0_skim.root',
        'VHH_CV_1_C2V_1_kl_0_hbbhbb': 'Boosted/CV_1_0_C2V_1_0_C3_0_0_skim.root',
        'VHH_CV_1_C2V_1_kl_20_hbbhbb': 'Boosted/CV_1_0_C2V_1_0_C3_20_0_skim.root',
}
branch_names = [
                'selLeptons_pt_0',
                'VHHFatJet1_Pt',
                'VHHFatJet2_Pt',
                'VHHFatJet1_Msoftdrop',
                'VHHFatJet2_Msoftdrop',
                'VHHFatJet_mjj',
                'VHHFatJet_HHPt',
                'selLeptons_phi_0',
                'VHHFatJet1_phi',
                'VHHFatJet2_phi',
                ]
ttbbstitching = '&& ( (sampleIndex==202||sampleIndex==203||sampleIndex==212)*(IsttB==0) || (sampleIndex==24||sampleIndex==25||sampleIndex==26)*(IsttB>0) || (sampleIndex<0))'

#Wenu >0.9
passcut = "isWenu && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9"+ttbbstitching
bkgfailcut = "isWenu && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)"+ttbbstitching
flattencut = "(isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9"+ttbbstitching
weightfile='test_1025_B_datacards_signalAll/extractWeight_validation_TT_0p9_1.root'
outputfile=OutDir+'datacards_Wenu_svb_signalAll_0p9_1.root'
makedatacards(outputfile, passcut , bkgfailcut, flattencut, weightfile, 'histsweight',procs_rew, filename,branch_names,bdt_svb,bdt_rew,True)
#plots_proce = [["TT_PassedBKG","TT_FailReweight"], ["TTB_PassedBKG","TTB_FailReweight"]]
#plot_BKGcomparison([outputfile], plots_proce)


#Wmunu >0.9
passcut = "isWmunu && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9"+ttbbstitching
bkgfailcut = "isWmunu && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)"+ttbbstitching
flattencut = "(isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9"+ttbbstitching
weightfile='test_1025_B_datacards_signalAll/extractWeight_validation_TT_0p9_1.root'
outputfile=OutDir+'datacards_Wmunu_svb_signalAll_0p9_1.root'
makedatacards(outputfile, passcut , bkgfailcut, flattencut, weightfile, 'histsweight',procs_rew, filename,branch_names,bdt_svb,bdt_rew,True)
#plots_proce = [["TT_PassedBKG","TT_FailReweight"], ["TTB_PassedBKG","TTB_FailReweight"]]
#plot_BKGcomparison([outputfile], plots_proce)
#exit()
#
##Wenu >0.94
#
passcut = "isWenu && VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94"+ttbbstitching
bkgfailcut = "isWenu && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)"+ttbbstitching
flattencut = "(isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94"+ttbbstitching
weightfile='test_1025_B_datacards_signalAll/extractWeight_validation_TT_0p94_1.root'
outputfile=OutDir+'datacards_Wenu_svb_signalAll_0p94_1.root'
makedatacards(outputfile, passcut , bkgfailcut, flattencut, weightfile, 'histsweight',procs_rew, filename,branch_names,bdt_svb1,bdt_rew1,True)
#plots_proce = [["TT_PassedBKG","TT_FailReweight"], ["TTB_PassedBKG","TTB_FailReweight"]]
#plot_BKGcomparison([outputfile], plots_proce)

##plots_proce = [["TT_PassedBKG","TT_FailReweight"], ["TTB_PassedBKG","TTB_FailReweight"]]
##plots_proce = [["TT_FailReweight","TT_FailReweight_up","TT_FailReweight_down"],["TTB_FailReweight","TTB_FailReweight_up","TTB_FailReweight_down"]]
##plot_BKGcomparison_updown([OutDir+"/extractWeight_and_Do_validation_all_0p94_Wenu_datacards_orginal.root"], plots_proce)

#Wmunu >0.94
passcut = "isWmunu && VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94"+ttbbstitching
bkgfailcut = "isWmunu && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)"+ttbbstitching
flattencut = "(isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94"+ttbbstitching
weightfile='test_1025_B_datacards_signalAll/extractWeight_validation_TT_0p94_1.root'
outputfile=OutDir+'datacards_Wmunu_svb_signalAll_0p94_1.root'
makedatacards(outputfile,passcut,bkgfailcut,flattencut, weightfile,'histsweight',procs_rew, filename,branch_names,bdt_svb1,bdt_rew1,True)
#plots_proce = [["TT_PassedBKG","TT_FailReweight"], ["TTB_PassedBKG","TTB_FailReweight"]]
#plot_BKGcomparison([outputfile], plots_proce)

##plots_proce = [["TT_PassedBKG","TT_FailReweight"], ["TTB_PassedBKG","TTB_FailReweight"]]
##plots_proce = [["TT_FailReweight","TT_FailReweight_up","TT_FailReweight_down"],["TTB_FailReweight","TTB_FailReweight_up","TTB_FailReweight_down"]]
##plot_BKGcomparison_updown([OutDir+"/extractWeight_and_Do_validation_all_0p94_Wmunu_datacards.root"], plots_proce)
###
##
##Wenu >0.9, <0.94
passcut = "isWenu && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9 && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)"+ttbbstitching
bkgfailcut = "isWenu && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)"+ttbbstitching
flattencut = "(isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9 && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)"+ttbbstitching
weightfile='test_1025_B_datacards_signalAll/extractWeight_validation_TT_0p9_0p94.root'
outputfile=OutDir+'datacards_Wenu_svb_signalAll_0p9_0p94.root'
makedatacards(outputfile, passcut , bkgfailcut, flattencut, weightfile, 'histsweight',procs_rew, filename,branch_names,bdt_svb2,bdt_rew2,False)
#plots_proce = [["TT_PassedBKG","TT_FailReweight"], ["TTB_PassedBKG","TTB_FailReweight"]]
#plot_BKGcomparison([outputfile], plots_proce)

msscut = "isWmunu && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9 && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)"+ttbbstitching
bkgfailcut = "isWmunu && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)"+ttbbstitching
flattencut = "(isWenu||isWmunu) && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9 && !(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)"+ttbbstitching
weightfile='test_1025_B_datacards_signalAll/extractWeight_validation_TT_0p9_0p94.root'
outputfile=OutDir+'datacards_Wmunu_svb_signalAll_0p9_0p94.root'
makedatacards(outputfile, passcut , bkgfailcut, flattencut, weightfile, 'histsweight',procs_rew, filename,branch_names,bdt_svb2,bdt_rew2,False)
#plots_proce = [["TT_PassedBKG","TT_FailReweight"], ["TTB_PassedBKG","TTB_FailReweight"]]
#plot_BKGcomparison([outputfile], plots_proce)
#
#
#
#
#
#
