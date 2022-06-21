#!/usr/bin/env python
import ROOT as r,sys,math,array,os
from optparse import OptionParser
from ROOT import std,RooDataHist #,RooMyPdf
from array import array
import numpy as np
from scipy.stats import poisson, norm, kstest
import pandas

import h5py

#from pvalue import *

fOutput="OutputFromMaker.root"
fHists=[]

numBins = 5
consideredFrac = .1

minM = 1800
maxM = 4500
#minM = 1900
#maxM = 2700
#minMForHists = 1600
#maxMForHists = 6000
minMForHists = 1460
maxMForHists = 6808
binForHists = 75

minMscan = 2200
maxMscan = 2700
massgaps = 100

sigMin = 75
sigMax = 100
sigGap = 25

bkgThreshold = [1000 for i in range(numBins+1)]

sigThresholds = [[] for i in range(numBins)]
for i in range(numBins):
    sigThresholds[i] = [1000 for b in range(numBins+1)]
    sigThresholds[i][0] = 0


def end():
    if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q','a',' ' ]:
            rep = input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]

#def drawFrame(iX,iData,iBkg,iFuncs,iCat):
def drawFrame(iX,iData,iFuncs,iCat):
    lCan   = r.TCanvas("qcd_"+iCat,"qcd_"+iCat,800,600)
    leg = r.TLegend(0.55,0.63,0.86,0.87)
    lFrame = iX.frame()
    lFrame.SetTitle("")
    lFrame.GetXaxis().SetTitle("m_{jj} (GeV)")
    lFrame.GetYaxis().SetTitle("Events")
    #iBkg.plotOn(lFrame,r.RooFit.FillColor(r.TColor.GetColor(100, 192, 232)),r.RooFit.FillStyle(3008), r.RooFit.DrawOption("E3"), r.RooFit.LineColor(r.kBlue))
    iData.plotOn(lFrame)
    iColor=51
    lRange = len(iFuncs)
    for i0 in range(lRange):
        if i0 == 1:
            iFuncs[i0].plotOn(lFrame,r.RooFit.LineColor(r.kGreen+1))
            chi_square = lFrame.chiSquare()
            print("CHI-SQUARED VALUE (Sig + Bkg): ", chi_square)
        else:
            iFuncs[i0].plotOn(lFrame,r.RooFit.LineColor(iColor),r.RooFit.LineStyle(r.kDashed))
            chi_square = lFrame.chiSquare()
            print("CHI-SQUARED VALUE (Bkg Only): ", chi_square)
        iColor+=10
    leg.SetFillColor(0)
    lFrame.Draw()
    lTmpData  = r.TH1F("tmpData" ,"tmpData" ,1,0,10); lTmpData .SetMarkerStyle(r.kFullCircle);
    #lTmpBkg   = r.TH1F("tmpBkg"  ,"tmpBkg"  ,1,0,10); lTmpBkg  .SetFillStyle(3008); lTmpBkg.SetLineColor(r.kBlue); lTmpBkg.SetFillColor(r.TColor.GetColor(100, 192, 232));
    lTmpFunc1 = r.TH1F("tmpFunc1","tmpFunc1",1,0,10); lTmpFunc1.SetLineColor(51);                lTmpFunc1.SetLineWidth(2); lTmpFunc1.SetLineStyle(r.kDashed);
    lTmpFunc2 = r.TH1F("tmpFunc2","tmpFunc2",1,0,10); lTmpFunc2.SetLineColor(61);                lTmpFunc2.SetLineWidth(2); lTmpFunc2.SetLineStyle(r.kDashed);
    lTmpFunc3 = r.TH1F("tmpFunc3","tmpFunc3",1,0,10); lTmpFunc3.SetLineColor(r.kGreen+1);        lTmpFunc3.SetLineWidth(2); #lTmpFunc3.SetLineStyle(r.kDashed);
    leg.AddEntry(lTmpData,"Toy Data","lpe")
    #leg.AddEntry(lTmpBkg ,"loss-sideband data","f")
    leg.AddEntry(lTmpFunc2,"Background-only Fit","lp")
    leg.AddEntry(lTmpFunc3,"Background+Signal Fit","lp")
    #leg.AddEntry(lTmpFunc1,"loss-sideband","lp")
    lPT = r.TPaveText(5200,20,6500,33)
    lPT.AddText("BlackBox 1")
    lPT.SetBorderSize(0)
    lPT.SetFillColor(0)
    leg.Draw()
    lCan.Modified()
    lCan.Update()
    lPT.Draw()
    lCan.SaveAs(lCan.GetName()+".pdf")
    end()

# build workspace
def workspace(iOutput,iDatas,iFuncs,iRates,iCat="cat0"):
    print('--- workspace')
    lW = r.RooWorkspace("w_"+str(iCat))
    for pData in iDatas:
        print('adding data ',pData,pData.GetName())
        getattr(lW,'import')(pData,r.RooFit.RecycleConflictNodes())
    for pFunc in iFuncs:
        print('adding func ',pFunc,pFunc.GetName())
        getattr(lW,'import')(pFunc,r.RooFit.RecycleConflictNodes())
    for pRate in iRates:
        print('adding rate ',pRate,pRate.GetName())
        getattr(lW,'import')(pRate,r.RooFit.RecycleConflictNodes())
    if iCat.find("pass_cat0") == -1:
        lW.writeToFile(iOutput,False)
    else:
        lW.writeToFile(iOutput)
    return lW

def clip(iData,iMin,iMax):
    pMinBin = 0
    pMaxBin = iData.GetNbinsX()
    for i0 in range(iData.GetNbinsX()+1):
        pLVal = iData.GetBinLowEdge(i0)
        pHVal = iData.GetBinLowEdge(i0)
        if iMin > pLVal:
            pMinBin = i0
        if iMax > pHVal:
            pMaxBin = i0
    NBins = pMaxBin-pMinBin
    pMinLow = iData.GetBinLowEdge(pMinBin)
    pMinMax = iData.GetBinLowEdge(pMaxBin)
    pData = r.TH1F(iData.GetName()+"R",iData.GetName()+"R",NBins,pMinLow,pMinMax)
    for i0 in range(NBins):
        print(iData.GetBinLowEdge(i0+pMinBin),pData.GetBinLowEdge(i0+1),"! Done")
        print(i0+1,iData.GetBinContent(i0+pMinBin))
        pData.SetBinContent(i0+1,iData.GetBinContent(i0+pMinBin))
    fHists.append(pData)
    return pData

#def fitFunc(iData,iBkg,iCat,iMin,iMax,iStep,iFixToSB=False):
def fitFunc(iData,iCat,iMin,iMax,iStep,iFixToSB=False):
    pData = clip(iData,iMin,iMax)
    #pBkg  = clip(iBkg ,iMin,iMax)
    #pBkg  = iBkg#clip(iBkg ,3000,6200)
    lXMin=pData.GetXaxis().GetXmin()
    lXMax=pData.GetXaxis().GetXmax()
    lNBins=pData.GetNbinsX()
    lX = r.RooRealVar("x","x",lXMin,lXMax)
    lX.setBins(lNBins)
    print("printing pData.Integral() = ",pData.Integral())
    lNTot   = r.RooRealVar("qcdnorm_"+iCat,"qcdnorm_"+iCat,pData.Integral(),0,3*pData.Integral())
    lA0     = r.RooRealVar   ("a0"+"_"+iCat,"a0"+"_"+iCat,0.0001,-1.,1.)
    lA1     = r.RooRealVar   ("a1"+"_"+iCat,"a1"+"_"+iCat,0.0001,-1,1.)
    lA2     = r.RooRealVar   ("a2"+"_"+iCat,"a2"+"_"+iCat,0.0001,-1,1)
    lA3     = r.RooRealVar   ("a3"+"_"+iCat,"a3"+"_"+iCat,0.0001,-1,1)
    lA4     = r.RooRealVar   ("a4"+"_"+iCat,"a4"+"_"+iCat,0.0001,-1,1)
    lA5     = r.RooRealVar   ("a5"+"_"+iCat,"a5"+"_"+iCat,0.0001,-1,1)
    lQFuncP = r.RooBernstein("tqcd_pass_"+iCat,"tqcd_pass_"+iCat,lX,r.RooArgList(lA0,lA1,lA2,lA3))#,lA5))
    #lQFuncP = r.RooBernstein("tqcd_pass_"+iCat,"tqcd_pass_"+iCat,lX,r.RooArgList(lA0,lA1,lA2,lA3,lA5))

    #lA0      = r.RooRealVar   ("a0"+"_"+iCat,"a0"+"_"+iCat,1.0,-200.,200.); #lA0.setConstant(r.kTRUE)
    #lA1      = r.RooRealVar   ("a1"+"_"+iCat,"a1"+"_"+iCat,1.00,-200.,200.)
    #lA2      = r.RooRealVar   ("a2"+"_"+iCat,"a2"+"_"+iCat,3.00,-200.,200.)
    #lQFuncP  = r.RooGenericPdf("tqcd_pass_"+iCat,"tqcd_pass_"+iCat,"(1-@0/13000.)**@2*(@1/13000.)**-@2",r.RooArgList(lX,lA1,lA2))#,lA5))
    lQCDP   = r.RooExtendPdf("qcd_"+iCat, "qcd"+iCat,lQFuncP,lNTot)

    #lBNTot   = r.RooRealVar("bqcdnorm_"+iCat,"bqcdnorm_"+iCat,pData.Integral(),0,3*pData.Integral())
    lBNTot   = r.RooRealVar("bkg_rate","bkg_rate",pData.Integral(),0,3*pData.Integral())
    #lBA0      = r.RooRealVar   ("ba0"+"_"+iCat,"ba0"+"_"+iCat,0.00,-200.,200.)
    #lBA1      = r.RooRealVar   ("ba1"+"_"+iCat,"ba1"+"_"+iCat,0.00,-200.,200.)
    #lBA2      = r.RooRealVar   ("ba2"+"_"+iCat,"ba2"+"_"+iCat,0.00,-200.,200.)
    lBA0     = r.RooRealVar   ("ba0"+"_"+iCat,"ba0"+"_"+iCat,0.00,-1.,1.)
    lBA1     = r.RooRealVar   ("ba1"+"_"+iCat,"ba1"+"_"+iCat,0.01,-1,1.)
    lBA2     = r.RooRealVar   ("ba2"+"_"+iCat,"ba2"+"_"+iCat,0.01,-1,1)
    lBA3     = r.RooRealVar   ("ba3"+"_"+iCat,"ba3"+"_"+iCat,0.01,-1,1)

    #lBQFuncP  = r.RooGenericPdf("btqcd_pass_"+iCat,"btqcd_pass_"+iCat,"(1-@0/13000.)**@1*(@0/13000.)**-@2",r.RooArgList(lX,lBA1,lBA2))
    lBQFuncP = r.RooBernstein("tqcd_pass_"+iCat,"tqcd_pass_"+iCat,lX,r.RooArgList(lBA0,lBA1,lBA2,lBA3))
    lBQCDP    = r.RooExtendPdf ("bqcd_"+iCat, "bqcd"+iCat,lBQFuncP,lBNTot)

    lMass   = r.RooRealVar("MH","MH",3500,1500,4500); lMass.setConstant(r.kTRUE)
    lSigma  = r.RooRealVar("sigma","Width of Gaussian",120,10,500); lSigma.setConstant(r.kTRUE)

    """
    lGaus   = r.RooGaussian("gauss","gauss(x,mean,sigma)",lX,lMass,lSigma)
    lNSig   = r.RooRealVar("signorm_"+iCat,"signorm_"+iCat,0.1*pData.Integral(),0,0.3*pData.Integral())
    lSig    = r.RooExtendPdf("sig_"+iCat, "sig_"+iCat,lGaus,lNSig)
    """
    lNSig   = r.RooRealVar("sig_rate","sig_rate",0.1*pData.Integral(),0,0.3*pData.Integral())
    lSig   = r.RooGaussian("sig","gauss(x,mean,sigma)",lX,lMass,lSigma)
    #lBkg   = r.RooBernstein("bkg","bernstein_for_QCD",lX,r.RooArgList(lA0,lA1,lA2,lA3))


    #biasDec = r.RooRealVar("biasDec","biasDec",0.00,-500,500)
    #sigmaDec = r.RooRealVar("sigmaDec","sigmaDec",1.00,-50,50)
    #tm = r.RooTruthModel("tm","truth model",lX)
    #gm = r.RooGaussModel("gm", "gauss model", lX, biasDec, sigmaDec)
    #lTau     = r.RooRealVar   ("tau"+"_"+iCat,"tau"+"_"+iCat,500,1,10000.)
    #lBkg = r.RooDecay("bkg", "decay_for_QCD", lX, lTau, tm, r.RooDecay.SingleSided)
    p0 = r.RooRealVar("p0"+"_"+iCat,"p0"+"_"+iCat,1,1,1)
    p1 = r.RooRealVar("p1"+"_"+iCat,"p1"+"_"+iCat,10.00,-100,100)
    p2 = r.RooRealVar("p2"+"_"+iCat,"p2"+"_"+iCat,4.00,0,100)
    p3 = r.RooRealVar("p3"+"_"+iCat,"p3"+"_"+iCat,1.05,-10,30)
    lBkg = r.RooMyPdf("bkg","dijet_decay",lX,p0,p1,p2,p3)
    #lBkg = r.RooDecay("bkg", "decay_for_QCD", lX, lTau, gm, r.RooDecay.SingleSided)


    #lBkg   = r.RooBernstein("bkg","bernstein_for_QCD",lX,r.RooArgList(lA0,lA1,lA2))
    #lBkg    = r.RooAbsPdf()
    lTot    = r.RooAddPdf("model", "model", r.RooArgList(lSig, lQCDP))
    lHData  = r.RooDataHist("data_obs","data_obs", r.RooArgList(lX),pData)
    #lHBkg   = r.RooDataHist("bkgestimate","bkgestimate", r.RooArgList(lX),pBkg)

    #lBQCDP.fitTo(lHBkg)
    
    #lQCDP.fitTo(lHData);
    #if iFixToSB:
    #    lA1.setConstant(r.kTRUE); lA2.setConstant(r.kTRUE);
    #lTot.fitTo(lHData,r.RooFit.Extended(r.kTRUE))
    
    #drawFrame(lX,lHData,[lQCDP,lTot],iCat)
    #print("!!!!!!!!!!!!!!!!!!!!!!! NSig",lNSig.getVal())

    masses =  array( 'd' )
    pvalues = array( 'd' )
    
    print("Creating datacard ..")
    dc_templ_file = open("./test_BINTEMPLATE.txt")
    dc_file = open("test_"+iCat+".txt","w")
    for line in dc_templ_file:
        line=line.replace('WORKSPACE', "w_"+iCat)
        line=line.replace('SIGMAX', str(0.3*pData.Integral()))
        line=line.replace('SIGINTEGRAL', str(0.1*pData.Integral()))
        line=line.replace('BKGMAX', str(3*pData.Integral()))
        line=line.replace('BKGINTEGRAL', str(pData.Integral()))
        line=line.replace('CUTS', str(iCat))
        #line=line.replace('OUTDIR', out_dir)                                                                                                                                                                                                            
        dc_file.write(line)
    dc_file.close()
    dc_templ_file.close()

    #lW = workspace(fOutput,[lHData],[lTot,lQCDP],iCat)
    lW = workspace(fOutput,[lHData],[lSig,lBkg], [lBNTot,lNSig],iCat)
    """
    lW.defineSet("poi","signorm_"+iCat)
    bmodel = r.RooStats.ModelConfig("b_model",lW)
    bmodel.SetPdf(lW.pdf("model"))
    bmodel.SetNuisanceParameters(r.RooArgSet(lA1,lA2,lA3,lNTot))
    bmodel.SetObservables(r.RooArgSet(lX))
    bmodel.SetParametersOfInterest(lW.set("poi"))
    lW.var("signorm_"+iCat).setVal(0)
    bmodel.SetSnapshot(lW.set("poi"))

    sbmodel = r.RooStats.ModelConfig("s_model",lW)
    sbmodel.SetPdf(lW.pdf("model"))
    sbmodel.SetNuisanceParameters(r.RooArgSet(lA1,lA2,lA3,lNTot))
    sbmodel.SetObservables(r.RooArgSet(lX))
    sbmodel.SetParametersOfInterest(lW.set("poi"))
    lW.var("signorm_"+iCat).setVal(lNSig.getVal())
    sbmodel.SetSnapshot(lW.set("poi"))
    
    stepsize = (iMax-iMin)/iStep
    masslist = [iMin + i*stepsize for i in range(iStep+1)]
    for mass in masslist:
        lW.var("mass").setVal(mass)
        ac = r.RooStats.AsymptoticCalculator(lHData, sbmodel, bmodel)
        ac.SetOneSidedDiscovery(True)
        ac.SetPrintLevel(-1)
        asResult = ac.GetHypoTest()
        pvalue=asResult.NullPValue()
        if pvalue > 1e-8:
            masses.append(mass)
            pvalues.append(pvalue)
            print(mass,pvalue)
    print(masses)
    print(pvalues)
    """
    return masses,pvalues

def setupData(iFileName):
    lDatas=[]
    lFile = r.TFile(iFileName)
    lH    = lFile.Get("data_obs")
    lH2   = lFile.Get("bkgestimate")

    lH.SetDirectory(0)
    lH2.SetDirectory(0)
    for i1 in range(lH.GetNbinsX()+1):
        lH.SetBinError(i1,math.sqrt(lH.GetBinContent(i1)))
        lH2.SetBinError(i1,math.sqrt(lH2.GetBinContent(i1)))
    lFile.Close()

    return lH, lH2

def sigVsMassPlot(masses,pvalues,labels):
    lC0 = r.TCanvas("A","A",800,600)
    leg = r.TLegend(0.55,0.23,0.86,0.47)
    leg.SetFillColor(0)
    lGraphs=[]
    sigmas=[]
    for i0 in range(len(masses)):
        graph1 = r.TGraph(len(masses[i0]),masses[i0],pvalues[i0])
        graph1.SetMarkerStyle(20)
        graph1.GetXaxis().SetTitle("m_{jj} (GeV)")
        graph1.GetYaxis().SetTitle("p^{0} value")
        graph1.SetTitle("")#Significance vs Mass")
        graph1.SetLineColor(51+i0*12)
        graph1.SetMarkerColor(51+i0*12)
        graph1.SetLineWidth(2+i0)
        r.gPad.SetLogy(True)
        graph1.GetYaxis().SetRangeUser(1e-8,1.0)
        if i0 == 0:
            graph1.Draw("alp")
        else:
            graph1.Draw("lp")
        lGraphs.append(graph1)
        leg.AddEntry(graph1,labels[i0],"lp")
    #sigmas=[0.317,0.045,0.0027,0.0000633721,0.0000005742]
    lines=[]
    for i0 in range(5):#len(sigmas)):
        sigmas.append(1-norm.cdf(i0+1))
        lLine = r.TLine(masses[0][0],sigmas[i0],masses[0][len(masses[0])-1],sigmas[i0])
        lLine.SetLineStyle(r.kDashed)
        lLine.SetLineWidth(2)
        lLine.Draw()
        lPT = r.TPaveText(3200,sigmas[i0],3700,sigmas[i0]+1.5*sigmas[i0])
        lPT.SetFillStyle(4050)
        lPT.SetFillColor(0)
        lPT.SetBorderSize(0)
        lPT.AddText(str(i0+1)+"#sigma")
        lPT.Draw()
        lines.append(lLine)
        lines.append(lPT)

    for pGraph in lGraphs:
        pGraph.Draw("lp")
    leg.Draw()
    lC0.Update()
    lC0.Draw()
    lC0.SaveAs("pvalue_bb1.png")
    end()

#def makeHist(iName,iCut,iBBTree,iBkgTree):
def makeHist(iName,iCut,iBBTree):
    lData1 = r.TH1F("bbhist"+iName,"bbhist"+iName,binForHists,minMForHists,maxMForHists)
    #lBkg1  = r.TH1F("bkhist"+iName,"bkhist"+iName,50,1200,6000)

    #cut="loss2 > 5.5 && loss1 < 10.0"

    print("loss2 > %s && loss2 < %s && loss1 > %s && loss1 < %s" % (str(bkgThreshold[iCut[0]]), str(bkgThreshold[iCut[0]+1]), str(sigThresholds[iCut[0]][iCut[1]]), str(sigThresholds[iCut[0]][iCut[1]+1])))
    
    iBBTree .Draw("mass>>bbhist"+iName,"loss2 > %s && loss2 < %s && loss1 > %s && loss1 < %s" % (str(bkgThreshold[iCut[0]]), str(bkgThreshold[iCut[0]+1]), str(sigThresholds[iCut[0]][iCut[1]]), str(sigThresholds[iCut[0]][iCut[1]+1])),"goff")
    
    #iBkgTree.Draw("mass>>bkhist"+iName,iCut)
    #lBkg1.Scale(lData1.Integral()/lBkg1.Integral())
    #return lData1,lBkg1
    fi = r.TFile(fOutput,"UPDATE")
    #f.cd()
    lData1.Write()
    fi.Close()
    return lData1

if __name__ == "__main__":#blackbox2-CutFromMap.root


    r.gSystem.Load('RooMyPdf_cxx.so')

    i = 1
    inputFileName = ""
    corner = False
    while i < len(sys.argv):
        if sys.argv[i] == "-f":
            inputFileName = str(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "-c":
            if(sys.argv[i+1]):
                corner = True
            i += 2
        else:
            print("Invalid input")
            i += 1

    eta_cut = None
    flow_type = 'NSQUAD'
    #flow_type = 'NSQUAD_nocontam'

    if not eta_cut:
        eta_str = 'None'
    else:
        eta_str = str(eta_cut)
        eta_str = eta_str.replace('.','p')


    #lTFile = r.TFile("./etacut%s_%s.root" % (eta_str, flow_type))
    lTFile = r.TFile("./"+inputFileName)
    lTTree = lTFile.Get("output")


    #output = r.RDataFrame("output", "./etacut%s_%s.root" % (eta_str, flow_type))
    #print(output.Count())
    #print(output.Filter("loss2>15").Count())

    lAll = r.TH1F("all","all",binForHists,minMForHists,maxMForHists)
    #iBBTree .Draw("mass>>bbhist"+iName,"loss2 > %s && loss2 < %s && loss1 > %s && loss1 < %s" % (str(bkgThreshold[iCut[0]]), str(bkgThreshold[iCut[0]+1]), str(sigThresholds[iCut[0]][iCut[1]]), str(sigThresholds[iCut[0]][iCut[1]+1])),"goff")
    lTTree.Draw("mass>>all","loss2 > 0","goff")
    TotalNum = lAll.Integral()
    print(TotalNum)

    #output = r.RDataFrame("output", "./etacut%s_%s.root" % (eta_str, flow_type))
    ##TotalNum = float(len(pandas.DataFrame(output.Filter("loss2>0").AsNumpy(columns=["loss2"]))["loss2"].to_numpy()))
    ##blah2 = pandas.DataFrame(blah)
    ##print(len(blah2["loss2"].to_numpy()))
    print(TotalNum)

    binFrac = consideredFrac/float(numBins)
    binFrac2D = binFrac/float(numBins)

    print(binFrac)
    print(binFrac2D)


    minCut = consideredFrac*TotalNum
    startCut = 0
    print(minCut)
    for b in range(numBins):
        print(b, minCut, binFrac, TotalNum, minCut - b*binFrac*TotalNum)
        for i in range(startCut,100):
            lTest = r.TH1F("testH","testH",binForHists,minMForHists,maxMForHists)
            lTTree.Draw("mass>>testH","loss2>%s" % (str(i)),"goff")
            passNum = lTest.Integral()
            #passNum = float(len(pandas.DataFrame(output.Filter("loss2>%s" % (str(i))).AsNumpy(columns=["loss2"]))["loss2"].to_numpy()))
            print(b, i, passNum)
            if(passNum < minCut - b*binFrac*TotalNum):
                #bkgThreshold[b] = i-1
                #startCut = i-1
                #break
                passed = False
                for j in range(10):
                    lTest2 = r.TH1F("testH2","testH2",binForHists,minMForHists,maxMForHists)
                    lTTree.Draw("mass>>testH2","loss2>%s" % (str(float(i-1)+0.1*j)),"goff")
                    passNum2 = lTest2.Integral()
                    #passNum2 = float(len(pandas.DataFrame(output.Filter("loss2>%s" % (str(float(i-1)+0.1*j))).AsNumpy(columns=["loss2"]))["loss2"].to_numpy()))
                    print(b, float(i-1)+0.1*j, passNum2)
                    if(passNum2 < minCut - b*binFrac*TotalNum):
                        bkgThreshold[b] = float(i-1)+0.1*(float(j)-1)
                        startCut = i-1
                        passed = True
                        break
                if(passed == False):
                    bkgThreshold[b] = float(i)
                    startCut = i-1
                break
    print(bkgThreshold)
    

    for b1 in range(numBins):
        startCut = 0
        for b2 in range(1,numBins):
            for i in range(startCut,100):
                lTest = r.TH1F("testH","testH",binForHists,minMForHists,maxMForHists)
                lTTree.Draw("mass>>testH","loss2>%s && loss2<%s && loss1>%s && loss1<%s" % (str(bkgThreshold[b1]), str(bkgThreshold[b1+1]), str(sigThresholds[b1][b2-1]), str(i)),"goff")
                passNum = lTest.Integral()
                #passNum = float(len(pandas.DataFrame(output.Filter("loss2>%s && loss2<%s && loss1>%s && loss1<%s" % (str(bkgThreshold[b1]), str(bkgThreshold[b1+1]), str(sigThresholds[b1][b2-1]), str(i))).AsNumpy(columns=["loss2"]))["loss2"].to_numpy()))
                print(binFrac2D*TotalNum, b1, b2, i, str(bkgThreshold[b1]), str(bkgThreshold[b1+1]), str(sigThresholds[b1][b2-1]), str(i), passNum)
                if(passNum > binFrac2D*TotalNum):
                    #sigThresholds[b1][b2] = i-1
                    #startCut = i-1
                    passed = False
                    for j in range(10):
                        lTest2 = r.TH1F("testH2","testH2",binForHists,minMForHists,maxMForHists)
                        lTTree.Draw("mass>>testH2","loss2>%s && loss2<%s && loss1>%s && loss1<%s" % (str(bkgThreshold[b1]), str(bkgThreshold[b1+1]), str(sigThresholds[b1][b2-1]), str(float(i)+0.1*j)),"goff")
                        passNum2 = lTest2.Integral()
                        #passNum2 = float(len(pandas.DataFrame(output.Filter("loss2>%s && loss2<%s && loss1>%s && loss1<%s" % (str(bkgThreshold[b1]), str(bkgThreshold[b1+1]), str(sigThresholds[b1][b2-1]), str(float(i)+0.1*j))).AsNumpy(columns=["loss2"]))["loss2"].to_numpy()))
                        if(passNum2 > binFrac2D*TotalNum):
                            sigThresholds[b1][b2] = float(i-1)+0.1*(float(j)-1)
                            startCut = i-1
                            passed = True
                            break
                    if(passed == False):
                        sigThresholds[b1][b2] = float(i)
                        startCut = i-1
                    break

    print(sigThresholds)


    df = r.RDataFrame("output", "./"+inputFileName)
    for b1 in range(numBins):
        for b2 in range(numBins):
            print("loss2>%s && loss2<%s && loss1>%s && loss1<%s" % (str(bkgThreshold[b1]), str(bkgThreshold[b1+1]), str(sigThresholds[b1][b2]), str(sigThresholds[b1][b2+1])))
            df_tmp = df.Filter("loss2>%s && loss2<%s && loss1>%s && loss1<%s" % (str(bkgThreshold[b1]), str(bkgThreshold[b1+1]), str(sigThresholds[b1][b2]), str(sigThresholds[b1][b2+1])))
            npy = df_tmp.AsNumpy(columns=["mass"])
            print(npy)
            print(len(npy['mass']))
            hf = h5py.File("bkgL_%s_%s_sigL_%s_%s" % (str(int(10*bkgThreshold[b1])), str(int(10*bkgThreshold[b1+1])), str(int(10*sigThresholds[b1][b2])), str(int(10*sigThresholds[b1][b2+1]))) + ".h5", 'w')
            hf.create_dataset('mjj', data=np.asarray(npy['mass']))
            hf.close()



    print("Creating combine command ..")
    dc_file = open("combineCommand.sh","w")
    dc_file.write("combineCards.py ")
    for i0 in range(numBins):
        if(corner and i0 < numBins-2):
            continue
        for i1 in range(numBins):
            if(corner and i1 > 1):
                continue
            title = "bkgL_%s_%s_sigL_%s_%s" % (str(int(10*bkgThreshold[i0])), str(int(10*bkgThreshold[i0+1])), str(int(10*sigThresholds[i0][i1])), str(int(10*sigThresholds[i0][i1+1])))
            dc_file.write("bin_"+title+"=datacard_JJ_"+title+"_FLOAT.txt ")

    dc_file.write("> fullCard.txt\n")
    dc_file.close()


    numprinted = 0
    cutNameList = '"'
    for i0 in range(numBins):
        if(corner and i0 < numBins-2):
            continue
        for i1 in range(numBins):
            if(corner and i1 > 1):
                continue
            cutNameList += "bkgL_%s_%s_sigL_%s_%s" % (str(int(10*bkgThreshold[i0])), str(int(10*bkgThreshold[i0+1])), str(int(10*sigThresholds[i0][i1])), str(int(10*sigThresholds[i0][i1+1])))
            if((not corner and i0*i1 < (numBins-1)*(numBins-1)) or (corner and numprinted < 3)):
                cutNameList += '" "'
                numprinted = numprinted+1
            else:
                cutNameList += '"'

    dc_templ_file = open("./significance_TEMPLATE.sh")
    dc_file = open("significance.sh","w")
    for line in dc_templ_file:
        line=line.replace('LISTOFFILES', cutNameList)
        line=line.replace('MASSMIN', str(minMscan))
        line=line.replace('MASSMAX', str(maxMscan))
        line=line.replace('GAPS', str(massgaps))
        dc_file.write(line)
    dc_file.close()
    dc_templ_file.close()

    
"""
    bkg_loss_list = [15,10]
    sig_loss_list = [25,20]

    #bkg_loss_list.reverse()
    sig_loss_list.reverse()

    label=[[] for i in range(numBins)]

    cuts=[]

    for b1 in range(numBins):
        for b2 in range(numBins):
            label[b1].append("L_{bkg} > %s && L_{bkg} < %s && L_{sig} > %s && L_{sig} < %s" % (str(int(10*bkgThreshold[b1])), str(int(10*bkgThreshold[b1+1])), str(int(10*sigThresholds[b1][b2])), str(int(10*sigThresholds[b1][b2+1]))))
            #label.append("L_{bkg} > %s   && L_{sig} < %s" % (str(bkg_loss), str(sig_loss)))
        

    for bkg_loss in bkg_loss_list:
        for sig_loss in sig_loss_list:
            #cuts.append("loss2 > %s   && loss1 < %s" % (str(bkg_loss), str(sig_loss)))
            cuts.append([bkg_loss,sig_loss])

    masses=[]
    pvalues=[]

    iBkgTemp=False
    print(len(cuts))
"""

"""
    for i0 in range(len(cuts)):
        #pData1,pBkg1=makeHist(label[i0],cuts[i0],lTTree,lBkgTree)
        pData1=makeHist(label[i0],cuts[i0],lTTree)
        #masses1,pvalues1=fitFunc(pData1,pBkg1,"BB1",1500,4500,200,iBkgTemp)
        masses1,pvalues1=fitFunc(pData1,"bkgL_%s_sigL_%s" % (str(cuts[i0][0]), str(cuts[i0][1])),1500,4500,200,iBkgTemp)
        pvalues.append(pvalues1)
        masses.append(masses1)
    #sigVsMassPlot(masses,pvalues,label)
"""

"""
    for i0 in range(numBins):
        for i1 in range(numBins):
            pData1=makeHist(label[i0][i1],[i0,i1],lTTree)
            #masses1,pvalues1=fitFunc(pData1,pBkg1,"BB1",1500,4500,200,iBkgTemp)
            masses1,pvalues1=fitFunc(pData1,"bkgL_%s_%s_sigL_%s_%s" % (str(int(10*bkgThreshold[i0])), str(int(10*bkgThreshold[i0+1])), str(int(10*sigThresholds[i0][i1])), str(int(10*sigThresholds[i0][i1+1]))),minM,maxM,200,iBkgTemp)
            pvalues.append(pvalues1)
            masses.append(masses1)

    
    print("Creating combine command ..")
    #dc_templ_file = open("./test_BINTEMPLATE.txt")
    dc_file = open("combineCommand.txt","w")
    dc_file.write("combineCards.py ")
    for i0 in range(numBins):
        for i1 in range(numBins):
            title = "bkgL_%s_%s_sigL_%s_%s" % (str(int(10*bkgThreshold[i0])), str(int(10*bkgThreshold[i0+1])), str(int(10*sigThresholds[i0][i1])), str(int(10*sigThresholds[i0][i1+1])))
            dc_file.write("bin_"+title+"=test_"+title+".txt ")

    dc_file.write("> fullCard.txt\n")
    for i0 in range(numBins):
        for i1 in range(numBins):
            title = "bkgL_%s_%s_sigL_%s_%s" % (str(int(10*bkgThreshold[i0])), str(int(10*bkgThreshold[i0+1])), str(int(10*sigThresholds[i0][i1])), str(int(10*sigThresholds[i0][i1+1])))
            dc_file.write("text2workspace.py test_"+title+".txt\n")
    dc_file.close()

    #dc_file = open("checkCommand.txt","w")
    #dc_file.write("combine -M ChannelCompatibilityCheck fullCard.root -m 2500 --preFitValue 0 --rMax 1000 --setParameterRanges sigma=75,75")
    combComm = "combine -M ChannelCompatibilityCheck fullCard.root -m ${mass} --preFitValue 0 --rMax 1000 --setParameterRanges sigma=${sig},${sig}"
    for i0 in range(numBins):
        for i1 in range(numBins):
            title = "bkgL_%s_%s_sigL_%s_%s" % (str(int(10*bkgThreshold[i0])), str(int(10*bkgThreshold[i0+1])), str(int(10*sigThresholds[i0][i1])), str(int(10*sigThresholds[i0][i1+1])))
            combComm = combComm+":sig_rate_"+title+"=1,1"
            #dc_file.write(":sig_rate_"+title+"=1,1")

    combComm = combComm+" --saveFitResult --saveNLL --fixedSignalStrength 0"

    #dc_file.write(" --saveFitResult --saveNLL --fixedSignalStrength 0")
    #dc_file.close()
    #combine -M ChannelCompatibilityCheck fullCard.root -m 2500 --preFitValue 0 --rMax 1000 --setParameterRanges sigma=75,75:sig_rate_bkgL_190_10000_sigL_389_10000=1,1:sig_rate_bkgL_162_190_sigL_299_10000=1,1:sig_rate_bkgL_162_190_sigL_0_299=1,1:sig_rate_bkgL_190_10000_sigL_0_389=1,1 --saveFitResult
"""
