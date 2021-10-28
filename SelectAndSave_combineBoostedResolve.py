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

# This script do two things
# 1. select branches from branch_names, we can use these branches to do training 
# 2. after training, we get the BDT model, we can add the BDT output to the root files 


def skimrootfile(inputfile, outstring, passcut, branch_names):
    test = root2array(inputfile, "Events", branches=branch_names, selection=passcut)
    print('len ', len(test))
    array2root(test, outstring,treename='Events', mode='recreate')

    inputpicklefile_R_c3_20vs0 = 'test_1024_R_c3BDT/BDT_c3_20vs0_BDTAdaBoost.pk'
    inputpicklefile_R_c3_20vs1 = 'test_1024_R_c3BDT/BDT_c3_20vs1_BDTAdaBoost.pk'
    inputpicklefile_R_c3_20vsc2v_2 = 'test_1024_R_c3BDT/BDT_c3_20vsc2v_2_BDTAdaBoost.pk'
    bdt_R_c3_20vs0 = pl.load(open(inputpicklefile_R_c3_20vs0,"rb"))
    bdt_R_c3_20vs1 = pl.load(open(inputpicklefile_R_c3_20vs1,"rb"))
    bdt_R_c3_20vsc2v_2 = pl.load(open(inputpicklefile_R_c3_20vsc2v_2,"rb"))
    add_BDT2rootfile=True
    if add_BDT2rootfile:
        #Load data
        branchlist = [
                'V_pt','VHH_H1_m','VHH_H1_e','VHH_H1_pT','VHH_H1_eta','VHH_H2_m','VHH_H2_e','VHH_H2_pT','VHH_H2_eta',
                'VHH_HH_e','VHH_HH_m',
                'VHH_HH_eta','VHH_HH_deta','VHH_HH_dphi','VHH_V_H2_dPhi','VHH_HH_dR','VHH_H2H1_pt_ratio'
                        ]
        test = root2array(inputfile, "Events", branchlist,selection=passcut)
        test = rec2array(test)
        RBDT_c3_20vs0 = bdt_R_c3_20vs0.decision_function(pd.DataFrame(test))
        RBDT_c3_20vs0 = np.array(RBDT_c3_20vs0 , dtype=[('RBDT_c3_20vs0', np.float64)])
        rnp.array2root(RBDT_c3_20vs0, outstring , 'Events')
        RBDT_c3_20vs1 = bdt_R_c3_20vs1.decision_function(pd.DataFrame(test))
        RBDT_c3_20vs1 = np.array(RBDT_c3_20vs1 , dtype=[('RBDT_c3_20vs1', np.float64)])
        rnp.array2root(RBDT_c3_20vs1, outstring , 'Events')
        RBDT_c3_20vsc2v_2 = bdt_R_c3_20vsc2v_2.decision_function(pd.DataFrame(test))
        RBDT_c3_20vsc2v_2 = np.array(RBDT_c3_20vsc2v_2 , dtype=[('RBDT_c3_20vsc2v_2', np.float64)])
        rnp.array2root(RBDT_c3_20vsc2v_2, outstring , 'Events')

    return


def selectandsave_RBDT(filestring, outstring, passcut, branch_names):
    #Load BDT
    add_BDT2rootfile=False
    if add_BDT2rootfile:
        inputpicklefile_R_c3_20vs0 = 'test_1024_R_c3BDT/BDT_c3_20vs0_BDTAdaBoost.pk'
        inputpicklefile_R_c3_20vs1 = 'test_1024_R_c3BDT/BDT_c3_20vs1_BDTAdaBoost.pk'
        inputpicklefile_R_c3_20vsc2v_2 = 'test_1024_R_c3BDT/BDT_c3_20vsc2v_2_BDTAdaBoost.pk'
        bdt_R_c3_20vs0 = pl.load(open(inputpicklefile_R_c3_20vs0,"rb"))
        bdt_R_c3_20vs1 = pl.load(open(inputpicklefile_R_c3_20vs1,"rb"))
        bdt_R_c3_20vsc2v_2 = pl.load(open(inputpicklefile_R_c3_20vsc2v_2,"rb"))
        #Load data
        branchlist = [
                'V_pt','VHH_H1_m','VHH_H1_e','VHH_H1_pT','VHH_H1_eta','VHH_H2_m','VHH_H2_e','VHH_H2_pT','VHH_H2_eta',
                'VHH_HH_e','VHH_HH_m',
                'VHH_HH_eta','VHH_HH_deta','VHH_HH_dphi','VHH_V_H2_dPhi','VHH_HH_dR','VHH_H2H1_pt_ratio'
                        ]
        test = root2array(filestring, "Events", branchlist)
        test = rec2array(test)
        print('len ', len(test))
        RBDT_c3_20vs0 = bdt_R_c3_20vs0.decision_function(pd.DataFrame(test))
        RBDT_c3_20vs0 = np.array(RBDT_c3_20vs0 , dtype=[('RBDT_c3_20vs0', np.float64)])
        rnp.array2root(RBDT_c3_20vs0, outstring , 'Events')
        RBDT_c3_20vs1 = bdt_R_c3_20vs1.decision_function(pd.DataFrame(test))
        RBDT_c3_20vs1 = np.array(RBDT_c3_20vs1 , dtype=[('RBDT_c3_20vs1', np.float64)])
        rnp.array2root(RBDT_c3_20vs1, outstring , 'Events')
        RBDT_c3_20vsc2v_2 = bdt_R_c3_20vsc2v_2.decision_function(pd.DataFrame(test))
        RBDT_c3_20vsc2v_2 = np.array(RBDT_c3_20vsc2v_2 , dtype=[('RBDT_c3_20vsc2v_2', np.float64)])
        rnp.array2root(RBDT_c3_20vsc2v_2, outstring , 'Events')

    add_BDT2rootfile=True
    if add_BDT2rootfile:
        inputpicklefile_R_nocat_c3_20 = 'test_1025_R_svbBDT/BDT_R_svb_noc3bdt_signalc3_20_BDTAdaBoost.pk'
        bdt_R_nocat_c3_20 = pl.load(open(inputpicklefile_R_nocat_c3_20,"rb"))
        inputpicklefile_R_nocat_c2v_2 = 'test_1025_R_svbBDT/BDT_R_svb_noc3bdt_signalc2v_2_BDTAdaBoost.pk'
        bdt_R_nocat_c2v_2 = pl.load(open(inputpicklefile_R_nocat_c2v_2,"rb"))
        inputpicklefile_R_nocat_SM = 'test_1025_R_svbBDT/BDT_R_svb_noc3bdt_signalSM_BDTAdaBoost.pk'
        bdt_R_nocat_SM = pl.load(open(inputpicklefile_R_nocat_SM,"rb"))

        inputpicklefile_R_usec3bdt_c3_20vs0_H = 'test_1025_R_svbBDT/BDT_R_svb_usec3bdt_c3_20vs0_H_BDTAdaBoost.pk'
        bdt_R_usec3bdt_c3_20vs0_H = pl.load(open(inputpicklefile_R_usec3bdt_c3_20vs0_H,"rb"))
        inputpicklefile_R_usec3bdt_c3_20vs0_L = 'test_1025_R_svbBDT/BDT_R_svb_usec3bdt_c3_20vs0_L_BDTAdaBoost.pk'
        bdt_R_usec3bdt_c3_20vs0_L = pl.load(open(inputpicklefile_R_usec3bdt_c3_20vs0_L,"rb"))
        inputpicklefile_R_usec3bdt_c3_20vs1_H = 'test_1025_R_svbBDT/BDT_R_svb_usec3bdt_c3_20vs1_H_BDTAdaBoost.pk'
        bdt_R_usec3bdt_c3_20vs1_H = pl.load(open(inputpicklefile_R_usec3bdt_c3_20vs1_H,"rb"))
        inputpicklefile_R_usec3bdt_c3_20vs1_L = 'test_1025_R_svbBDT/BDT_R_svb_usec3bdt_c3_20vs1_L_BDTAdaBoost.pk'
        bdt_R_usec3bdt_c3_20vs1_L = pl.load(open(inputpicklefile_R_usec3bdt_c3_20vs1_L,"rb"))
        inputpicklefile_R_usec3bdt_c3_20vsc2v_2_H = 'test_1025_R_svbBDT/BDT_R_svb_usec3bdt_c3_20vsc2v_2_H_BDTAdaBoost.pk'
        bdt_R_usec3bdt_c3_20vsc2v_2_H = pl.load(open(inputpicklefile_R_usec3bdt_c3_20vsc2v_2_H,"rb"))
        inputpicklefile_R_usec3bdt_c3_20vsc2v_2_L = 'test_1025_R_svbBDT/BDT_R_svb_usec3bdt_c3_20vsc2v_2_L_BDTAdaBoost.pk'
        bdt_R_usec3bdt_c3_20vsc2v_2_L = pl.load(open(inputpicklefile_R_usec3bdt_c3_20vsc2v_2_L,"rb"))

        #Load data
        branchlist = [
                    'j1_btagcat','j2_btagcat','j3_btagcat','j4_btagcat',
                    'selLeptons_pt_0',
                    'VHH_H1_pT',
                    'VHH_H2_pT',
                    'VHH_H1_m',
                    'VHH_H2_m',
                    'VHH_HH_m',
                    'VHH_HH_pT',
                    'selLeptons_phi_0',
                    'VHH_H1_phi',
                    'VHH_H2_phi',
                        ]
        test = root2array(filestring, "Events", branchlist)
        test = rec2array(test)
        print('len ', len(test))
        RsVbBDT_nocat_c3_20 = bdt_R_nocat_c3_20.decision_function(pd.DataFrame(test))
        RsVbBDT_nocat_c3_20 = np.array(RsVbBDT_nocat_c3_20 , dtype=[('RsVbBDT_nocat_c3_20', np.float64)])
        rnp.array2root(RsVbBDT_nocat_c3_20, outstring , 'Events')
        RsVbBDT_nocat_c2v_2 = bdt_R_nocat_c2v_2.decision_function(pd.DataFrame(test))
        RsVbBDT_nocat_c2v_2 = np.array(RsVbBDT_nocat_c2v_2 , dtype=[('RsVbBDT_nocat_c2v_2', np.float64)])
        rnp.array2root(RsVbBDT_nocat_c2v_2, outstring , 'Events')
        RsVbBDT_nocat_SM = bdt_R_nocat_SM.decision_function(pd.DataFrame(test))
        RsVbBDT_nocat_SM = np.array(RsVbBDT_nocat_SM , dtype=[('RsVbBDT_nocat_SM', np.float64)])
        rnp.array2root(RsVbBDT_nocat_SM, outstring , 'Events')
        RsVbBDT_cat_c3_20vs0_H = bdt_R_usec3bdt_c3_20vs0_H.decision_function(pd.DataFrame(test))
        RsVbBDT_cat_c3_20vs0_H = np.array(RsVbBDT_cat_c3_20vs0_H , dtype=[('RsVbBDT_cat_c3_20vs0_H', np.float64)])
        rnp.array2root(RsVbBDT_cat_c3_20vs0_H, outstring , 'Events')
        RsVbBDT_cat_c3_20vs0_L = bdt_R_usec3bdt_c3_20vs0_L.decision_function(pd.DataFrame(test))
        RsVbBDT_cat_c3_20vs0_L = np.array(RsVbBDT_cat_c3_20vs0_L , dtype=[('RsVbBDT_cat_c3_20vs0_L', np.float64)])
        rnp.array2root(RsVbBDT_cat_c3_20vs0_L, outstring , 'Events')
        RsVbBDT_cat_c3_20vs1_H = bdt_R_usec3bdt_c3_20vs1_H.decision_function(pd.DataFrame(test))
        RsVbBDT_cat_c3_20vs1_H = np.array(RsVbBDT_cat_c3_20vs1_H , dtype=[('RsVbBDT_cat_c3_20vs1_H', np.float64)])
        rnp.array2root(RsVbBDT_cat_c3_20vs1_H, outstring , 'Events')
        RsVbBDT_cat_c3_20vs1_L = bdt_R_usec3bdt_c3_20vs1_L.decision_function(pd.DataFrame(test))
        RsVbBDT_cat_c3_20vs1_L = np.array(RsVbBDT_cat_c3_20vs1_L , dtype=[('RsVbBDT_cat_c3_20vs1_L', np.float64)])
        rnp.array2root(RsVbBDT_cat_c3_20vs1_L, outstring , 'Events')
        RsVbBDT_cat_c3_20vsc2v_2_H = bdt_R_usec3bdt_c3_20vsc2v_2_H.decision_function(pd.DataFrame(test))
        RsVbBDT_cat_c3_20vsc2v_2_H = np.array(RsVbBDT_cat_c3_20vsc2v_2_H , dtype=[('RsVbBDT_cat_c3_20vsc2v_2_H', np.float64)])
        rnp.array2root(RsVbBDT_cat_c3_20vsc2v_2_H, outstring , 'Events')
        RsVbBDT_cat_c3_20vsc2v_2_L = bdt_R_usec3bdt_c3_20vsc2v_2_L.decision_function(pd.DataFrame(test))
        RsVbBDT_cat_c3_20vsc2v_2_L = np.array(RsVbBDT_cat_c3_20vsc2v_2_L , dtype=[('RsVbBDT_cat_c3_20vsc2v_2_L', np.float64)])
        rnp.array2root(RsVbBDT_cat_c3_20vsc2v_2_L, outstring , 'Events')

def selectandsave_BBDT(filestring, outstring, passcut, branch_names):

    #Load BDT
    inputpicklefile_svb_gt_0p9_lt_0p94='../bdt_reweight_clean/test_1005_trainSvB_withRewFail/WlnB_svb_gt_0p9_lt_0p94__BDTAdaBoost.pk'
    inputpicklefile_svb_gt_0p94='../bdt_reweight_clean/test_1005_trainSvB_withRewFail/WlnB_svb_gt_0p94__BDTAdaBoost.pk'
    inputpicklefile_svb_gt_0p9='../bdt_reweight_clean/test_1005_trainSvB_withRewFail/WlnB_svb_gt_0p9__BDTAdaBoost.pk'
    inputpicklefile_rew_0p9_0p94='../bdt_reweight_clean/test_1003_gt0p94_trainreweight/AllB_TTAll_kine_rew_0p9_0p94__BDTAdaBoost.pk'
    inputpicklefile_rew_0p94='../bdt_reweight_clean/test_1003_gt0p94_trainreweight/AllB_TTAll_kine_rew_0p94__BDTAdaBoost.pk'
    inputpicklefile_rew_0p9='../bdt_reweight_clean/test_1005_trainSvB_withRewFail/AllB_TTAll_kine_rew_0p9__BDTAdaBoost.pk'

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

    passcut_0p9=' VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9'
    passcut_0p94='  VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94'
    passcut_0p9_0p94=' VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9 &&!(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)'
    failcut=' !( VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)'

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


#For boosted analysis 
branch_names = [
            'selLeptons_pt_0',
            'VHHFatJet1_Pt','VHHFatJet1VdPhi','VHHFatJet2VdPhi','V_pt',
            'VHHFatJet2_Pt','VHH_V_phi',
            'VHHFatJet1_Msoftdrop',
            'VHHFatJet2_Msoftdrop',
            'VHHFatJet_mjj',
            'VHHFatJet_HHPt',
            'selLeptons_phi_0',
            'VHHFatJet1_phi','IsttB',
            'VHHFatJet2_phi',
            'weight','isZee','isZmm','isZnn',
            'isWenu','isWmunu','VHHFatJet1_ParticleNetMD_bbvsQCD','VHHFatJet2_ParticleNetMD_bbvsQCD',
            'VHHFatJet_HHe',
            'h1_pnetcat','h2_pnetcat','MET_Pt','MET_Phi','lepMetDPhi','sampleIndex',
            'j1_btagcat','j2_btagcat','j3_btagcat','j4_btagcat',
            'VHH_H1_m','VHH_H1_e','VHH_H1_pT','VHH_H1_eta','VHH_H2_m','VHH_H2_e','VHH_H2_pT','VHH_H2_eta',
            'VHH_HH_e','VHH_HH_m','VHH_H1_phi','VHH_H2_phi','VHH_HH_pT','VHH_HH_phi','CMS_vhh_bdt_c2v_13TeV',
            'VHH_HH_eta','VHH_HH_deta','VHH_HH_dphi','VHH_V_H2_dPhi','VHH_HH_dR','VHH_H2H1_pt_ratio','VHH_rHH',
            ]
branch_names = [
            'VHH_H1_m','VHH_H1_pT','VHH_H1_eta','VHH_H1_phi','VHH_H1_e',
            'VHH_H2_m','VHH_H2_pT','VHH_H2_eta','VHH_H2_phi','VHH_H2_e',
            'VHH_HH_m','VHH_HH_pT','VHH_HH_eta','VHH_HH_phi','VHH_HH_e',
            'j1_btagcat','j2_btagcat','j3_btagcat','j4_btagcat',
            'lepMetDPhi','sampleIndex','CMS_vhh_bdt_c2v_13TeV',
            'VHH_HH_dphi','VHH_HH_deta','VHH_HH_dR','VHH_rHH','VHH_HT','IsttB',
            'VHH_mass','VHH_Vreco4j_HT','VHH_vsptmassRatio','VHH_H2H1_pt_ratio','VHH_V_HH_dPhi','VHH_V_H1_dPhi','VHH_V_H2_dPhi',
            'VHH_V_phi','VHH_V_m','VHH_V_e','VHH_nBJets','VHH_nBJets_loose','VHH_nBJets_tight','MET_Pt','MET_Phi','V_pt',
            'weight','isZnn','isZmm','isZee','selLeptons_pt_0','selLeptons_phi_0','selLeptons_eta_0',
            'isWenu','isWmunu',
            'isResolved','isBoosted',
            'VHHFatJet1_Pt','VHHFatJet1VdPhi','VHHFatJet2VdPhi',
            'VHHFatJet2_Pt',
            'VHHFatJet1_Msoftdrop',
            'VHHFatJet2_Msoftdrop',
            'VHHFatJet_mjj',
            'VHHFatJet_HHPt',
            'VHHFatJet1_phi',
            'VHHFatJet2_phi',
            'VHHFatJet1_ParticleNetMD_bbvsQCD','VHHFatJet2_ParticleNetMD_bbvsQCD',
            'VHHFatJet_HHe',
            'h1_pnetcat','h2_pnetcat',
            ]

passcut_0p9='isBoosted && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9'
passcut_0p94='isBoosted && VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94'
passcut_0p9_0p94='isBoosted && VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9 &&!(VHHFatJet1_ParticleNetMD_bbvsQCD>0.94 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.94)'
failcut='isBoosted && !( VHHFatJet1_ParticleNetMD_bbvsQCD>0.9 && VHHFatJet2_ParticleNetMD_bbvsQCD>0.9)'

Filelist = ['TT_']
Filelist = ['CV_1_0_C2V_0_0_C3_1_0','CV_0_5_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_0_0','CV_1_0_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_2_0','CV_1_0_C2V_2_0_C3_1_0']

Filelist = ['Run2018_EG_ReMiniAOD','Run2018_Mu_ReMiniAOD']

Filelist = ['CV_1_0_C2V_0_0_C3_1_0','CV_0_5_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_0_0','NLO_CV_1_0_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_2_0','CV_1_0_C2V_2_0_C3_1_0','CV_1_5_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_20_0']

#Filelist = ['TT_','ttbb','Run2018_EG_ReMiniAOD','Run2018_Mu_ReMiniAOD','CV_1_0_C2V_0_0_C3_1_0','CV_0_5_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_0_0','NLO_CV_1_0_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_2_0','CV_1_0_C2V_2_0_C3_1_0','CV_1_5_C2V_1_0_C3_1_0','CV_1_0_C2V_1_0_C3_20_0']
#Filelist = ['TT_','ttbb_']
#Filelist = ['TT_']

Filelist = ['TT_AllHadronic','TT_DiLep_NoPSWeights','TT_SingleLep']

#Filelist = ['TTttbb_']

doskim=True
doBbdt=False
doRbdt=False

for ifile in Filelist:
    passcut = '(isWenu||isWmunu) && ((isResolved && VHH_rHH>=0 && VHH_rHH<=50 && VHH_nBJets>2) || (isBoosted))'
    if doskim:
        inputfile='../TEST_1016_UL_newpost/*'+ifile+'*/*.root'
        outstring=ifile+'_skim.root'
        skimrootfile(inputfile, outstring, passcut, branch_names)
        if 'TT_' in ifile:
            for i in range(10):
                #inputfile='../TEST_1016_UL_newpost/*'+ifile+'*/*.root'
                inputfile='../TEST_1016_UL_newpost/'+ifile+'/*'+str(i)+'.root'
                print(inputfile)
                outstring=ifile+'_'+str(i)+'_skim.root'
                skimrootfile(inputfile, outstring, passcut, branch_names)
    if doBbdt:
        inputfile=ifile+'_skim.root'
        outstring=ifile+'_skim_addBDT.root'
        selectandsave_BBDT(inputfile, outstring, passcut, branch_names)
    if doRbdt:
        inputfile='Resolved_SR/'+ifile+'_skim.root'
        outstring='Resolved_SR/'+ifile+'_skim.root'
        selectandsave_RBDT(inputfile, outstring, passcut, branch_names)




