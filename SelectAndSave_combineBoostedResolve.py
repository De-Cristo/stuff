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
import sys
import os
import awkward as ak
import ROOT as R
import root_numpy as rnp
import bisect
import glob

# This script do two things
# 1. select branches from branch_names, we can use these branches to do training 
# 2. after training, we get the BDT model, we can add the BDT output to the root files 

def createDir(OutDir):
    if OutDir is not None :
        if not os.path.isdir( OutDir ) :
            os.makedirs( OutDir )

def skimrootfile(inputfile, outstring, passcut, branch_names):
    test = root2array(inputfile, "Events", branches=branch_names, selection=passcut)
    print('len ', len(test))
    array2root(test, outstring,treename='Events', mode='recreate')

    add_BDT2rootfile=False
    if add_BDT2rootfile:
        inputpicklefile_R_c3_20vs0 = 'test_1117_RBDT_cate_updateSF/RBDT_cat_20vs0_BDTAdaBoost.pk'
        #inputpicklefile_R_c3_20vs1 = 'test_1024_R_c3BDT/BDT_c3_20vs1_BDTAdaBoost.pk'
        #inputpicklefile_R_c3_20vsc2v_2 = 'test_1024_R_c3BDT/BDT_c3_20vsc2v_2_BDTAdaBoost.pk'
        bdt_R_c3_20vs0 = pl.load(open(inputpicklefile_R_c3_20vs0,"rb"))
        #bdt_R_c3_20vs1 = pl.load(open(inputpicklefile_R_c3_20vs1,"rb"))
        #bdt_R_c3_20vsc2v_2 = pl.load(open(inputpicklefile_R_c3_20vsc2v_2,"rb"))

        #Load data
        branchlist = [
                'V_pt','VHH_H1_m','VHH_H1_e','VHH_H1_pT','VHH_H1_eta','VHH_H2_m','VHH_H2_e','VHH_H2_pT','VHH_H2_eta',
                'VHH_HH_e','VHH_HH_m','VHH_HH_pT',
                'VHH_HH_eta','VHH_HH_deta','VHH_HH_dphi','VHH_V_H2_dPhi','VHH_HH_dR','VHH_H2H1_pt_ratio'
                        ]
        test = root2array(inputfile, "Events", branchlist,selection=passcut)
        test = rec2array(test)
        RBDT_c3_20vs0 = bdt_R_c3_20vs0.decision_function(pd.DataFrame(test))
        RBDT_c3_20vs0 = np.array(RBDT_c3_20vs0 , dtype=[('RBDT_cate_kl20vs0_1117', np.float64)])
        rnp.array2root(RBDT_c3_20vs0, outstring , 'Events')
        #RBDT_c3_20vs1 = bdt_R_c3_20vs1.decision_function(pd.DataFrame(test))
        #RBDT_c3_20vs1 = np.array(RBDT_c3_20vs1 , dtype=[('RBDT_c3_20vs1', np.float64)])
        #rnp.array2root(RBDT_c3_20vs1, outstring , 'Events')
        #RBDT_c3_20vsc2v_2 = bdt_R_c3_20vsc2v_2.decision_function(pd.DataFrame(test))
        #RBDT_c3_20vsc2v_2 = np.array(RBDT_c3_20vsc2v_2 , dtype=[('RBDT_c3_20vsc2v_2', np.float64)])
        #rnp.array2root(RBDT_c3_20vsc2v_2, outstring , 'Events')

    add_BDT2rootfile=False
    if add_BDT2rootfile:
        inputpicklefile_B_c3_20vsSM = 'test_1029_B_c3BDT/BDT_c3_20vsSM_BDTAdaBoost.pk'
        bdt_R_c3_20vsSM = pl.load(open(inputpicklefile_B_c3_20vsSM,"rb"))
        inputpicklefile_B_c3_20vs0 = 'test_1029_B_c3BDT/BDT_c3_20vsc3_0_BDTAdaBoost.pk'
        bdt_R_c3_20vs0 = pl.load(open(inputpicklefile_B_c3_20vs0,"rb"))
        #Load data
        branchlist = [
                'V_pt','VHHFatJet1_Msoftdrop','VHHFatJet1_e','VHHFatJet1_Pt','VHHFatJet1_eta','VHHFatJet1_phi',
                 'VHHFatJet2_Msoftdrop','VHHFatJet2_e','VHHFatJet2_Pt','VHHFatJet2_eta','VHHFatJet2_phi',
                'VHHFatJet_mjj','VHHFatJet_HHe','VHHFatJet_HHPt','VHHFatJet_HHeta','VHHFatJet_HHdR','VHHFatH1H2dPhi',
                'VHHFatJet1VdPhi','VHHFatJet1VdEta','VHHFatJet2VdPhi',
                ]
        test = root2array(inputfile, "Events", branchlist,selection=passcut)
        test = rec2array(test)
        #BBDT_catBDT_c3_20vsSM = bdt_R_c3_20vsSM.decision_function(pd.DataFrame(test))
        #BBDT_catBDT_c3_20vsSM = np.array(BBDT_catBDT_c3_20vsSM , dtype=[('BBDT_catBDT_c3_20vsSM', np.float64)])
        #rnp.array2root(BBDT_catBDT_c3_20vsSM, outstring , 'Events')
        BBDT_catBDT_c3_20vs0 = bdt_R_c3_20vs0.decision_function(pd.DataFrame(test))
        BBDT_catBDT_c3_20vs0 = np.array(BBDT_catBDT_c3_20vs0 , dtype=[('BBDT_catBDT_c3_20vs0', np.float64)])
        rnp.array2root(BBDT_catBDT_c3_20vs0, outstring , 'Events')
    return


def selectandsave_RBDT(filestring, outstring, datayear):
    #Load BDT

    add_CateBDT2rootfile=True
    add_SvBBDT2rootfile_wln=False
    add_SvBBDT2rootfile_znn=False
    #Cate. BDT
    if add_CateBDT2rootfile:
        inputpicklefile_R_c3_20vs0 = 'RBDT_cate_220120_run2mc_lessInput_4wln/RBDT_cat_20vs0_0117_wln_vpt_BDTAdaBoost.pk'
        inputpicklefile_R_c3_20vs0 = 'RBDT_cate_220120_run2mc_lessInput_4znn/RBDT_cat_20vs0_0117_znn_vpt_BDTAdaBoost.pk'
        bdt_R_c3_20vs0 = pl.load(open(inputpicklefile_R_c3_20vs0,"rb"))
        #Load data
        branchlist = [
                'V_pt','VHH_H1_m','VHH_H1_e','VHH_H1_pT','VHH_H2_m','VHH_H2_e','VHH_H2_pT',
                'VHH_HH_e','VHH_HH_m','VHH_HH_pT',
                'VHH_HH_eta','VHH_HH_deta','VHH_HH_dphi','VHH_V_H2_dPhi','VHH_HH_dR','VHH_H2H1_pt_ratio',datayear]
#        branchlist = [
#                'V_pt','VHH_H1_m','VHH_H1_e','VHH_H1_pT','VHH_H2_m','VHH_H2_e','VHH_H2_pT',
#                'VHH_HH_e','VHH_HH_m','VHH_HH_pT',
#                'VHH_HH_eta','VHH_HH_deta','VHH_HH_dphi','VHH_V_H2_dPhi','VHH_HH_dR','VHH_H2H1_pt_ratio','(isWenu)+(isWmunu)*2+(isZee)*3+(isZmm)*4+(isZnn)*5',datayear
#                        ]
        test = root2array(filestring, "Events", branches=branchlist)
        test = rec2array(test)
        print('len ', len(test))
        RBDT_c3_20vs0 = bdt_R_c3_20vs0.decision_function(pd.DataFrame(test))
        RBDT_c3_20vs0 = np.array(RBDT_c3_20vs0 , dtype=[('RBDT_c3_20vs0_0120_znn', np.float64)])
        rnp.array2root(RBDT_c3_20vs0, outstring , 'Events',mode='update')

    # SvB BDT wln
    if add_SvBBDT2rootfile_wln:
        inputpicklefile_Wlnsvb_h = 'RBDT_SvB_220117_run2mc/wlnRBDT_SvB_usekl20vs0_use_vpt_HighScore_BDTAdaBoost.pk'
        bdt_wlnsvb_H = pl.load(open(inputpicklefile_Wlnsvb_h,"rb"))
        inputpicklefile_Wlnsvb_l = 'RBDT_SvB_220117_run2mc/wlnRBDT_SvB_usekl20vs0_use_vpt_LowScore_BDTAdaBoost.pk'
        bdt_wlnsvb_L = pl.load(open(inputpicklefile_Wlnsvb_l,"rb"))
        #Load data
        branchlist = [
                   'VHH_H1_BJet1_btag','VHH_H1_BJet2_btag','VHH_H2_BJet1_btag','VHH_H2_BJet2_btag',
                    'V_pt', 'VHH_H1_pT','VHH_H2_pT',
                    'VHH_H1_m','VHH_H2_m','VHH_HH_m','VHH_HH_pT',
                    'VHH_V_phi','VHH_H1_phi','VHH_H2_phi',datayear
                        ]
        test = root2array(filestring, "Events", branches=branchlist)
        test = rec2array(test)
        print('len ', len(test))
        bdt_wlnsvb_H_ = bdt_wlnsvb_H.decision_function(pd.DataFrame(test))
        bdt_wlnsvb_H_ = np.array(bdt_wlnsvb_H_ , dtype=[('wlnRBDT_svb_kl20vs0_High_0119', np.float64)])
        rnp.array2root(bdt_wlnsvb_H_, outstring , 'Events', mode='update')
        bdt_wlnsvb_L_ = bdt_wlnsvb_L.decision_function(pd.DataFrame(test))
        bdt_wlnsvb_L_ = np.array(bdt_wlnsvb_L_ , dtype=[('wlnRBDT_svb_kl20vs0_Low_0119', np.float64)])
        rnp.array2root(bdt_wlnsvb_L_, outstring , 'Events', mode='update')

    # SvB BDT znn
    if add_SvBBDT2rootfile_znn:
        inputpicklefile_znnsvb_h = 'RBDT_SvB_220117_run2mc/znnRBDT_SvB_usekl20vs0_use_vpt_HighScore_BDTAdaBoost.pk'
        bdt_znnsvb_H = pl.load(open(inputpicklefile_znnsvb_h,"rb"))
        inputpicklefile_znnsvb_l = 'RBDT_SvB_220117_run2mc/znnRBDT_SvB_usekl20vs0_use_vpt_LowScore_BDTAdaBoost.pk'
        bdt_znnsvb_L = pl.load(open(inputpicklefile_znnsvb_l,"rb"))
        #Load data
        branchlist = [
                   'VHH_H1_BJet1_btag','VHH_H1_BJet2_btag','VHH_H2_BJet1_btag','VHH_H2_BJet2_btag',
                    'V_pt', 'VHH_H1_pT','VHH_H2_pT',
                    'VHH_H1_m','VHH_H2_m','VHH_HH_m','VHH_HH_pT',
                    'VHH_V_phi','VHH_H1_phi','VHH_H2_phi',datayear
                        ]
        test = root2array(filestring, "Events", branches=branchlist)
        test = rec2array(test)
        print('len ', len(test))
        bdt_znnsvb_H_ = bdt_znnsvb_H.decision_function(pd.DataFrame(test))
        bdt_znnsvb_H_ = np.array(bdt_znnsvb_H_ , dtype=[('znnRBDT_svb_kl20vs0_High_0119', np.float64)])
        rnp.array2root(bdt_znnsvb_H_, outstring , 'Events',mode='update')
        bdt_znnsvb_L_ = bdt_znnsvb_L.decision_function(pd.DataFrame(test))
        bdt_znnsvb_L_ = np.array(bdt_znnsvb_L_ , dtype=[('znnRBDT_svb_kl20vs0_Low_0119', np.float64)])
        rnp.array2root(bdt_znnsvb_L_, outstring , 'Events',mode='update')


def selectandsave_BBDT(filestring, outstring, passcut, branch_names):

    #Load BDT
    inputpicklefile_svb_gt_0p9_lt_0p94='test_1025_B_svb/WlnB_signalall_svb_0p9_0p94_BDTAdaBoost.pk'
    inputpicklefile_svb_gt_0p94='test_1025_B_svb/WlnB_signalall_svb_0p94_1_BDTAdaBoost.pk'
    inputpicklefile_svb_gt_0p9='test_1025_B_svb/WlnB_signalall_svb_0p9_1_BDTAdaBoost.pk'
    inputpicklefile_rew_0p9_0p94='reweightBDT/AllB_TTAll_kine_rew_0p9_0p94__BDTAdaBoost.pk'
    inputpicklefile_rew_0p94='reweightBDT/AllB_TTAll_kine_rew_0p94__BDTAdaBoost.pk'
    inputpicklefile_rew_0p9='reweightBDT/AllB_TTAll_kine_rew_0p9__BDTAdaBoost.pk'

    bdt_svb_0p9_0p94 = pl.load(open(inputpicklefile_svb_gt_0p9_lt_0p94,"rb"))
    bdt_svb_0p94 = pl.load(open(inputpicklefile_svb_gt_0p94,"rb"))
    bdt_svb_0p9 = pl.load(open(inputpicklefile_svb_gt_0p9,"rb"))
    bdt_rew_0p9_0p94 = pl.load(open(inputpicklefile_rew_0p9_0p94,"rb"))
    bdt_rew_0p94 = pl.load(open(inputpicklefile_rew_0p94,"rb"))
    bdt_rew_0p9 = pl.load(open(inputpicklefile_rew_0p9,"rb"))

    # add new cloumn called BDT_svb_gt_0p9_lt_0p94, BDT_svb_gt_0p94, BDT_svb_gt_0p9, BDT_rew_0p9_0p94, BDT_rew_0p94, BDT_rew_0p9
    add_BDT2rootfile=True
    if add_BDT2rootfile:
        #Load data
        branchlist = [
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
        test = root2array(filestring, "Events", branchlist)
        test = rec2array(test)
        print('len ', len(test))
        test_BDTsvb_0p9_0p94 = bdt_svb_0p9_0p94.decision_function(pd.DataFrame(test))
        test_BDTsvb_0p94 = bdt_svb_0p94.decision_function(pd.DataFrame(test))
        test_BDTsvb_0p9 = bdt_svb_0p9.decision_function(pd.DataFrame(test))
        test_BDTrew_0p9_0p94 = bdt_rew_0p9_0p94.decision_function(pd.DataFrame(test))
        test_BDTrew_0p94 = bdt_rew_0p94.decision_function(pd.DataFrame(test))
        test_BDTrew_0p9 = bdt_rew_0p9.decision_function(pd.DataFrame(test))

        #test_BDTsvb_0p9_0p94 = np.array(test_BDTsvb_0p9_0p94 , dtype=[('BDT_svb_gt_0p9_lt_0p94', np.float64)])
        ##rnp.array2root(test_BDTsvb_0p9_0p94, outstring , 'Events')
        #test_BDTsvb_0p94 = np.array(test_BDTsvb_0p94 , dtype=[('BDT_svb_gt_0p94', np.float64)])
        ##rnp.array2root(test_BDTsvb_0p94, outstring , 'Events')
        #test_BDTsvb_0p9 = np.array(test_BDTsvb_0p9 , dtype=[('BDT_svb_gt_0p9', np.float64)])
        ##rnp.array2root(test_BDTsvb_0p9, outstring , 'Events')
        #test_BDTrew_0p9_0p94 = np.array(test_BDTrew_0p9_0p94 , dtype=[('BDT_rew_gt_0p9_lt_0p94', np.float64)])
        ##rnp.array2root(test_BDTrew_0p9_0p94, outstring , 'Events')
        #test_BDTrew_0p94 = np.array(test_BDTrew_0p94 , dtype=[('BDT_rew_gt_0p94', np.float64)])
        ##rnp.array2root(test_BDTrew_0p94, outstring , 'Events')
        #test_BDTrew_0p9 = np.array(test_BDTrew_0p9 , dtype=[('BDT_rew_gt_0p9', np.float64)])
        ##rnp.array2root(test_BDTrew_0p9, outstring , 'Events')
        ##return
        #exit()

    passcut_0p9='isBoosted&& VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9'
    passcut_0p94='isBoosted&&  VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94'
    passcut_0p9_0p94='isBoosted&& VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9 &&!(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)'
    failcut='isBoosted&& !( VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)'

    do_flattening=True
    if do_flattening:
        branchlist = [
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
        bkgpass_0p94 = root2array(filestring, "Events", branchlist,selection=passcut_0p94)
        bkgpass_0p94 = rec2array(bkgpass_0p94)
        bkgpass_0p9 = root2array(filestring, "Events", branchlist,selection=passcut_0p9)
        bkgpass_0p9 = rec2array(bkgpass_0p9)
        bkgpass_0p9_0p94 = root2array(filestring, "Events", branchlist,selection=passcut_0p9_0p94)
        bkgpass_0p9_0p94 = rec2array(bkgpass_0p9_0p94)
        bkgfail = root2array(filestring, "Events", branchlist,selection=failcut)
        bkgfail = rec2array(bkgfail)
        print(len(bkgfail))

        bkgfail_rewBDT_0p9_0p94 = bdt_rew_0p9_0p94.decision_function(pd.DataFrame(bkgfail))
        bkgfail_rewBDT_0p94 = bdt_rew_0p94.decision_function(pd.DataFrame(bkgfail))
        bkgfail_rewBDT_0p9 = bdt_rew_0p9.decision_function(pd.DataFrame(bkgfail))

        bkgpass_0p9_0p94_rewBDT_0p9_0p94 = bdt_rew_0p9_0p94.decision_function(pd.DataFrame(bkgpass_0p9_0p94))
        bkgpass_0p9_rewBDT_0p9 = bdt_rew_0p9.decision_function(pd.DataFrame(bkgpass_0p9))
        bkgpass_0p94_rewBDT_0p94 = bdt_rew_0p94.decision_function(pd.DataFrame(bkgpass_0p94))
        if 'TT' in filestring or 'ttbb' in filestring:
            weight40p9_0p94 = getweight(bkgpass_0p9_0p94_rewBDT_0p9_0p94,bkgfail_rewBDT_0p9_0p94,test_BDTrew_0p9_0p94)
            weight40p9 = getweight(bkgpass_0p9_rewBDT_0p9,bkgfail_rewBDT_0p9,test_BDTrew_0p9)
            weight40p94 = getweight(bkgpass_0p94_rewBDT_0p94,bkgfail_rewBDT_0p94,test_BDTrew_0p94)
            print(len(weight40p9_0p94))
            print(len(weight40p9))
            print(len(test_BDTrew_0p9))
        else:
            weight40p9_0p94 = np.ones(len(test_BDTrew_0p9_0p94))
            weight40p9 = np.ones(len(test_BDTrew_0p9))
            weight40p94 = np.ones(len(test_BDTrew_0p94))

        test_BDTsvb_0p9_0p94 = np.array(test_BDTsvb_0p9_0p94 , dtype=[('BDT_svb_gt_0p9_lt_0p94', np.float64)])
        test_BDTsvb_0p94 = np.array(test_BDTsvb_0p94 , dtype=[('BDT_svb_gt_0p94', np.float64)])
        test_BDTsvb_0p9 = np.array(test_BDTsvb_0p9 , dtype=[('BDT_svb_gt_0p9', np.float64)])
        test_BDTrew_0p9_0p94 = np.array(test_BDTrew_0p9_0p94 , dtype=[('BDT_rew_gt_0p9_lt_0p94', np.float64)])
        test_BDTrew_0p94 = np.array(test_BDTrew_0p94 , dtype=[('BDT_rew_gt_0p94', np.float64)])
        test_BDTrew_0p9 = np.array(test_BDTrew_0p9 , dtype=[('BDT_rew_gt_0p9', np.float64)])
        weight40p9_0p94 = np.array(weight40p9_0p94 , dtype=[('weight4_0p9_0p94', np.float64)])
        weight40p9 = np.array(weight40p9 , dtype=[('weight4_0p9', np.float64)])
        weight40p94 = np.array(weight40p94 , dtype=[('weight4_0p94', np.float64)])

        rnp.array2root(test_BDTsvb_0p9_0p94, outstring , 'Events')
        rnp.array2root(test_BDTsvb_0p94, outstring , 'Events')
        rnp.array2root(test_BDTsvb_0p9, outstring , 'Events')
        rnp.array2root(test_BDTrew_0p9_0p94, outstring , 'Events')
        rnp.array2root(test_BDTrew_0p94, outstring , 'Events')
        rnp.array2root(test_BDTrew_0p9, outstring , 'Events')
        rnp.array2root(weight40p9_0p94, outstring , 'Events')
        rnp.array2root(weight40p9, outstring , 'Events')
        rnp.array2root(weight40p94, outstring , 'Events')



def getweight(bkgpass,bkgfail_rewBDT, bkgtest):
        print("============ Flatten passed reweight_BDT  ===========")
        bbins_passrewBDTflatten = []
        if len(bbins_passrewBDTflatten) == 0:
            print("============ find best binning ===========")
            decisions = bkgpass
            low = min(np.min(d) for d in decisions)
            high = max(np.max(d) for d in decisions)
            low_high = (low,high)
            print("low_high",low_high)
            low=-1
            high=1
            # --------------------------------------   get flat binning
            s_tot = np.ones(len(decisions)).sum()
            bins = 50
            values = np.sort(decisions)
            cumu = np.cumsum( np.ones(len(decisions))  )
            targets = [n*s_tot/bins for n in range(1,bins)]
            workingPoints = []
            for target in targets:
                index = np.argmax(cumu>target)
                workingPoints.append(values[index])
            bbins_passrewBDTflatten = [float(low)]+workingPoints+[float(high)]
        print(bbins_passrewBDTflatten)
        rew_weight = []
        for nele in range(len(bbins_passrewBDTflatten)-1):
            a=len(bkgfail_rewBDT[ (bkgfail_rewBDT>=bbins_passrewBDTflatten[nele]) & (bkgfail_rewBDT<bbins_passrewBDTflatten[nele+1]) ])
            #totl+=a
            #print('from ', bbins_passrewBDTflatten[nele],' to ',bbins_passrewBDTflatten[nele+1])#, ' is ',a,totl)
            b=len(bkgpass[ (bkgpass>=bbins_passrewBDTflatten[nele]) & (bkgpass<bbins_passrewBDTflatten[nele+1]) ])
            #print(a, b, b/a)
            rew_weight.append(b/a)
        print(rew_weight)
        if len(bbins_passrewBDTflatten) != len(rew_weight)+1:
            print('we got problems bbins_passrewBDTflatten has len ',len(bbins_passrewBDTflatten), ' but rew_weight has len ',len(rew_weight))
            exit()
        bkgfail_weight = [ rew_weight[int(bisect.bisect_left(bbins_passrewBDTflatten, x)-1)] for x in bkgtest]
        return bkgfail_weight



branch_names = [
            'VHH_H1_m','VHH_H1_pT','VHH_H1_eta','VHH_H1_phi','VHH_H1_e',
            'VHH_H2_m','VHH_H2_pT','VHH_H2_eta','VHH_H2_phi','VHH_H2_e',
            'VHH_HH_m','VHH_HH_pT','VHH_HH_eta','VHH_HH_phi','VHH_HH_e',
            'VHH_H1_BJet1_btag','VHH_H1_BJet2_btag','VHH_H2_BJet1_btag','VHH_H2_BJet2_btag',
            'j1_btagcat','j2_btagcat','j3_btagcat','j4_btagcat',
            'lepMetDPhi','sampleIndex',
            'VHH_HH_dphi','VHH_HH_deta','VHH_HH_dR','VHH_rHH','VHH_HT','IsttB',
            'VHH_mass','VHH_Vreco4j_HT','VHH_vsptmassRatio','VHH_H2H1_pt_ratio','VHH_V_HH_dPhi','VHH_V_H1_dPhi','VHH_V_H2_dPhi',
            'VHH_V_phi','VHH_V_m','VHH_V_e','VHH_nBJets','VHH_nBJets_loose','VHH_nBJets_tight','MET_Pt','MET_Phi','V_pt',
            'weight','isZnn','isZmm','isZee','selLeptons_pt_0','selLeptons_phi_0','selLeptons_eta_0',
            'isWenu','isWmunu',
            'isResolved','isBoosted',
            'VHHFatJet1_Msoftdrop','VHHFatJet1_Pt','VHHFatJet1_eta','VHHFatJet1_phi','VHHFatJet1_e',
            'VHHFatJet2_Msoftdrop','VHHFatJet2_Pt','VHHFatJet2_eta','VHHFatJet2_phi','VHHFatJet2_e',            
            'VHHFatJet_mjj','VHHFatJet_HHPt','VHHFatJet_HHe',
            'VHHFatJet1VdPhi','VHHFatJet2VdPhi','VHHFatH1H2dPhi',
            'VHHFatJet1_ParticleNetMD_bbvsQCD','VHHFatJet2_ParticleNetMD_bbvsQCD',
            'VHHFatJet_HHdR','VHHFatJet_HHeta','VHHFatJet1VdEta',
            'h1_pnetcat','h2_pnetcat','Pass_nominal','VHH_nFatJet','controlSample',
            #'wlnBBDT_rew_HMP','wlnBBDT_rew_LP','wlnBBDT_svb_HMP','wlnBBDT_svb_LP',
            #'RBDT_cate_kl20VSkl0','wlnRBDT_svb_Highscore','wlnRBDT_svb_Lowscore','znnRBDT_svb_Lowscore','znnRBDT_svb_Highscore'
            'wlnBBDT_SvB_lep_HMP_v2','wlnBBDT_SvB_lep_LP_v2','wlnBBDT_SvB_v_HMP_v2','wlnBBDT_SvB_v_LP_v2',
            'wlnBBDT_rew_lep_HMP_v2','wlnBBDT_rew_lep_LP_v2','wlnBBDT_rew_v_HMP_v2','wlnBBDT_rew_v_LP_v2',
            'wlnRBDT_cate_kl20VSkl0_v2','wlnRBDT_svb_lep_Highscore_v2','wlnRBDT_svb_lep_Lowscore_v2','wlnRBDT_svb_v_Highscore_v2','wlnRBDT_svb_v_Lowscore_v2',
            'znnBBDT_SvB_HMP_v2','znnBBDT_SvB_LP_v2','znnBBDT_rew_HMP_v2','znnBBDT_rew_LP_v2','znnRBDT_cate_kl20VSkl0_v2','znnRBDT_svb_Highscore_v2','znnRBDT_svb_Lowscore_v2'
            ]

passcut_0p9='isBoosted && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9'
passcut_0p94='isBoosted && VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94'
passcut_0p9_0p94='isBoosted && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9 &&!(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)'
failcut='isBoosted && !( VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)'

Filelist = ['TT_']
Filelist = ['CV_1_0_C2V_0_0_C3_1_0','CV_0_5_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_0_0','CV_1_0_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_2_0','CV_1_0_C2V_2_0_C3_1_0']

Filelist = ['Run2018_EG_ReMiniAOD','Run2018_Mu_ReMiniAOD','Run2018_MET_MiniAOD']

#Filelist = ['TT_','ttbb','Run2018_EG_ReMiniAOD','Run2018_Mu_ReMiniAOD','CV_1_0_C2V_0_0_C3_1_0','CV_0_5_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_0_0','NLO_CV_1_0_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_2_0','CV_1_0_C2V_2_0_C3_1_0','CV_1_5_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_20_0']

#Filelist = ['TT_AllHadronic','TT_DiLep_NoPSWeights','TT_SingleLep','TTBB_DiLep','TTBB_SingleLep','TTBB_AllHadronic']
#Filelist = ['TT_SingleLep']
#Filelist = ['TTBB_AllHadronic']
#Filelist = ['TT_DiLep_NoPSWeights','TT_SingleLep','TTBB_SingleLep']
#Filelist = ['WHHTo4B_CV_1_0_C2V_0_0_C3_1_0','WHHTo4B_CV_0_5_C2V_1_0_C3_1_0','WHHTo4B_CV_1_0_C2V_1_0_C3_0_0','WHHTo4B_CV_1_0_C2V_1_0_C3_1_0','WHHTo4B_CV_1_0_C2V_1_0_C3_2_0','WHHTo4B_CV_1_0_C2V_2_0_C3_1_0','WHHTo4B_CV_1_5_C2V_1_0_C3_1_0','WHHTo4B_CV_1_0_C2V_1_0_C3_20_0',  'ZHHTo4B_CV_1_0_C2V_0_0_C3_1_0']
#Filelist = ['ZHHTo4B_CV_0_5_C2V_1_0_C3_1_0','ZHHTo4B_CV_1_0_C2V_1_0_C3_0_0','ZHHTo4B_CV_1_0_C2V_1_0_C3_1_0','ZHHTo4B_CV_1_0_C2V_1_0_C3_2_0','ZHHTo4B_CV_1_0_C2V_2_0_C3_1_0','ZHHTo4B_CV_1_5_C2V_1_0_C3_1_0','ZHHTo4B_CV_1_0_C2V_1_0_C3_20_0',  ]
#Filelist = ['TT_AllHadronic','TT_DiLep_NoPSWeights','TT_SingleLep','TTBB_DiLep','TTBB_SingleLep','TTBB_AllHadronic','WHHTo4B_CV_1_0_C2V_0_0_C3_1_0','WHHTo4B_CV_0_5_C2V_1_0_C3_1_0','WHHTo4B_CV_1_0_C2V_1_0_C3_0_0','WHHTo4B_CV_1_0_C2V_1_0_C3_1_0','WHHTo4B_CV_1_0_C2V_1_0_C3_2_0','WHHTo4B_CV_1_0_C2V_2_0_C3_1_0','WHHTo4B_CV_1_0_C2V_1_0_C3_20_0',  'ZHHTo4B_CV_1_0_C2V_0_0_C3_1_0','ZHHTo4B_CV_0_5_C2V_1_0_C3_1_0','ZHHTo4B_CV_1_0_C2V_1_0_C3_0_0','ZHHTo4B_CV_1_0_C2V_1_0_C3_1_0','ZHHTo4B_CV_1_0_C2V_1_0_C3_2_0','ZHHTo4B_CV_1_0_C2V_2_0_C3_1_0','ZHHTo4B_CV_1_5_C2V_1_0_C3_1_0','ZHHTo4B_CV_1_0_C2V_1_0_C3_20_0','WHHTo4B_CV_1_5_C2V_1_0_C3_1_0']


doskim=True
doBbdt=False
doRbdt=False
doBoost=False

newdir='TEST_220122_2018_boosted_skim/'
createDir(newdir)
for ifile in Filelist:
    if doBoost:
        createDir(newdir+"/"+ifile)
        if 'Lep' in ifile:
            for i in range(10):
                inputfile=boosted_input+ifile+'/*'+str(i)+'.root'
                outstring=newdir+"/"+ifile+'/'+ifile+'_'+str(i)+'_skim.root'
                test = root2array(inputfile, "Events", branches=branch_names, selection='isBoosted')
                array2root(test, outstring,treename='Events', mode='recreate')
        else:
            inputfile=boosted_input+ifile+'*/*.root'
            outstring=newdir+"/"+ifile+'/'+ifile+'_skim.root'
            test = root2array(inputfile, "Events", branches=branch_names, selection='isBoosted')
            array2root(test, outstring,treename='Events', mode='recreate')
        continue

    if doskim:
        passcut = '(isBoosted)'
        #passcut = '(isResolved&&VHH_rHH>=0&&VHH_rHH<50&&VHH_nBJets>2)||(isBoosted)'
        createDir(newdir+"/"+ifile)
        #for iffile in glob.glob('../TEST_220108_2016pre/'+ifile+"/*"):
        #for iffile in glob.glob('../TEST_220120_2017_boosted/'+ifile+"/*"):
        i=0
        for iffile in glob.glob('../TEST_220122_2018/'+ifile+"/*"):
            i=i+1
            #if i<400: continue
            outstring=newdir+"/"+ifile+'/'+iffile.split('/')[-1].replace(".root","_skim.root")
            skimrootfile(iffile, outstring, passcut, branch_names)
    if doBbdt:
        inputfile='all/'+ifile+'_skim.root'
        outstring='all/'+ifile+'_skim.root'
        selectandsave_BBDT(inputfile, outstring, passcut, branch_names)
    if doRbdt:
        #createDir(newdir+"/"+ifile)
        for iffile in glob.glob(newdir+'/'+ifile+"/*"):
            outputfile=newdir+"/"+ifile+'/'+ifile+'_all.root'
            outputfile=iffile
            selectandsave_RBDT(iffile, outputfile,'2018')




