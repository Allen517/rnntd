package com.kingwang.rnntd.cell.impl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.rnntd.batchderv.BatchDerivative;
import com.kingwang.rnntd.batchderv.impl.LSTMBatchDerivative;
import com.kingwang.rnntd.cell.Operator;
import com.kingwang.rnntd.cell.RNNCell;
import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.utils.Activer;
import com.kingwang.rnntd.utils.LoadTypes;
import com.kingwang.rnntd.utils.MatIniter;
import com.kingwang.rnntd.utils.MatIniter.Type;

public class LSTM extends Operator implements RNNCell, Serializable  {
    private static final long serialVersionUID = -7059290852389115565L;
    
    private int outSize;
    
    /**
     * historical first-derivative gradient of weights
     */
    private DoubleMatrix hdWxi;
    private DoubleMatrix hdWdi;
    private DoubleMatrix hdWhi;
    private DoubleMatrix hdWci;
    private DoubleMatrix hdbi;
    
    private DoubleMatrix hdWxf;
    private DoubleMatrix hdWdf;
    private DoubleMatrix hdWhf;
    private DoubleMatrix hdWcf;
    private DoubleMatrix hdbf;
    
    private DoubleMatrix hdWxc;
    private DoubleMatrix hdWdc;
    private DoubleMatrix hdWhc;
    private DoubleMatrix hdbc;
    
    private DoubleMatrix hdWxo;
    private DoubleMatrix hdWdo;
    private DoubleMatrix hdWho;
    private DoubleMatrix hdWco;
    private DoubleMatrix hdbo;
    
    private DoubleMatrix hdWhd;
    private DoubleMatrix hdbd;
    
    private DoubleMatrix hdWhy;
    private DoubleMatrix hdby;
    
    /**
     * historical second-derivative gradient of weights
     */
    private DoubleMatrix hd2Wxi;
    private DoubleMatrix hd2Wdi;
    private DoubleMatrix hd2Whi;
    private DoubleMatrix hd2Wci;
    private DoubleMatrix hd2bi;
    
    private DoubleMatrix hd2Wxf;
    private DoubleMatrix hd2Wdf;
    private DoubleMatrix hd2Whf;
    private DoubleMatrix hd2Wcf;
    private DoubleMatrix hd2bf;
    
    private DoubleMatrix hd2Wxc;
    private DoubleMatrix hd2Wdc;
    private DoubleMatrix hd2Whc;
    private DoubleMatrix hd2bc;
    
    private DoubleMatrix hd2Wxo;
    private DoubleMatrix hd2Wdo;
    private DoubleMatrix hd2Who;
    private DoubleMatrix hd2Wco;
    private DoubleMatrix hd2bo;
    
    private DoubleMatrix hd2Whd;
    private DoubleMatrix hd2bd;
    
    private DoubleMatrix hd2Why;
    private DoubleMatrix hd2by;
    
    /**
     * network weights
     */
    public DoubleMatrix Wxi;
    public DoubleMatrix Wdi;
    public DoubleMatrix Whi;
    public DoubleMatrix Wci;
    public DoubleMatrix bi;
    
    public DoubleMatrix Wxf;
    public DoubleMatrix Wdf;
    public DoubleMatrix Whf;
    public DoubleMatrix Wcf;
    public DoubleMatrix bf;
    
    public DoubleMatrix Wxc;
    public DoubleMatrix Wdc;
    public DoubleMatrix Whc;
    public DoubleMatrix bc;
    
    public DoubleMatrix Wxo;
    public DoubleMatrix Wdo;
    public DoubleMatrix Who;
    public DoubleMatrix Wco;
    public DoubleMatrix bo;
    
    public DoubleMatrix Whd;
    public DoubleMatrix bd;
    
    public DoubleMatrix Why;
    public DoubleMatrix by;
    
    public LSTM(int inSize, int inTmSize, int outSize, int nodeSize, MatIniter initer) {
        this.outSize = outSize;
        
        if (initer.getType() == Type.Uniform) {
            this.Wxi = initer.uniform(inSize, outSize);
            this.Wdi = initer.uniform(inTmSize, outSize);
            this.Whi = initer.uniform(outSize, outSize);
            this.Wci = initer.uniform(outSize, outSize);
            this.bi = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxf = initer.uniform(inSize, outSize);
            this.Wdf = initer.uniform(inTmSize, outSize);
            this.Whf = initer.uniform(outSize, outSize);
            this.Wcf = initer.uniform(outSize, outSize);
            this.bf = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxc = initer.uniform(inSize, outSize);
            this.Wdc = initer.uniform(inTmSize, outSize);
            this.Whc = initer.uniform(outSize, outSize);
            this.bc = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxo = initer.uniform(inSize, outSize);
            this.Wdo = initer.uniform(inTmSize, outSize);
            this.Who = initer.uniform(outSize, outSize);
            this.Wco = initer.uniform(outSize, outSize);
            this.bo = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Whd = initer.uniform(outSize, nodeSize);
            this.bd = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
            
            this.Why = initer.uniform(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
            this.Wxi = initer.gaussian(inSize, outSize);
            this.Wdi = initer.gaussian(inTmSize, outSize);
            this.Whi = initer.gaussian(outSize, outSize);
            this.Wci = initer.gaussian(outSize, outSize);
            this.bi = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxf = initer.gaussian(inSize, outSize);
            this.Wdf = initer.gaussian(inTmSize, outSize);
            this.Whf = initer.gaussian(outSize, outSize);
            this.Wcf = initer.gaussian(outSize, outSize);
            this.bf = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxc = initer.gaussian(inSize, outSize);
            this.Wdc = initer.gaussian(inTmSize, outSize);
            this.Whc = initer.gaussian(outSize, outSize);
            this.bc = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxo = initer.gaussian(inSize, outSize);
            this.Wdo = initer.gaussian(inTmSize, outSize);
            this.Who = initer.gaussian(outSize, outSize);
            this.Wco = initer.gaussian(outSize, outSize);
            this.bo = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Whd = initer.gaussian(outSize, nodeSize);
            this.bd = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
            
            this.Why = initer.gaussian(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
            this.Wxi = initer.svd(inSize, outSize);
            this.Wdi = initer.svd(inTmSize, outSize);
            this.Whi = initer.svd(outSize, outSize);
            this.Wci = initer.svd(outSize, outSize);
            this.bi = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxf = initer.svd(inSize, outSize);
            this.Wdf = initer.svd(inTmSize, outSize);
            this.Whf = initer.svd(outSize, outSize);
            this.Wcf = initer.svd(outSize, outSize);
            this.bf = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxc = initer.svd(inSize, outSize);
            this.Wdc = initer.svd(inTmSize, outSize);
            this.Whc = initer.svd(outSize, outSize);
            this.bc = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxo = initer.svd(inSize, outSize);
            this.Wdo = initer.svd(inTmSize, outSize);
            this.Who = initer.svd(outSize, outSize);
            this.Wco = initer.svd(outSize, outSize);
            this.bo = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Whd = initer.svd(outSize, nodeSize);
            this.bd = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
            
            this.Why = initer.svd(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        	this.Wxi = DoubleMatrix.zeros(inSize, outSize).add(0.1);
        	this.Wdi = DoubleMatrix.zeros(inTmSize, outSize).add(0.1);
            this.Whi = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.Wci = DoubleMatrix.zeros(outSize, outSize).add(0.3);
            this.bi = DoubleMatrix.zeros(1, outSize).add(0.4);
            
            this.Wxf = DoubleMatrix.zeros(inSize, outSize).add(0.1);
            this.Wdf = DoubleMatrix.zeros(inTmSize, outSize).add(0.1);
            this.Whf = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.Wcf = DoubleMatrix.zeros(outSize, outSize).add(0.3);
            this.bf = DoubleMatrix.zeros(1, outSize).add(0.4);
            
            this.Wxc = DoubleMatrix.zeros(inSize, outSize).add(0.1);
            this.Wdc = DoubleMatrix.zeros(inTmSize, outSize).add(0.1);
            this.Whc = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.bc = DoubleMatrix.zeros(1, outSize).add(0.3);
            
            this.Wxo = DoubleMatrix.zeros(inSize, outSize).add(0.1);
            this.Wdo = DoubleMatrix.zeros(inTmSize, outSize).add(0.1);
            this.Who = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.Wco = DoubleMatrix.zeros(outSize, outSize).add(0.3);
            this.bo = DoubleMatrix.zeros(1, outSize).add(0.4);
            
            this.Whd = DoubleMatrix.zeros(outSize, nodeSize).add(.1);
            this.bd = new DoubleMatrix(1, nodeSize).add(.2);
        }
        this.hdWxi = new DoubleMatrix(inSize, outSize);
        this.hdWdi = new DoubleMatrix(inTmSize, outSize);
        this.hdWhi = new DoubleMatrix(outSize, outSize);
        this.hdWci = new DoubleMatrix(outSize, outSize);
        this.hdbi = new DoubleMatrix(1, outSize);
        
        this.hd2Wxi = new DoubleMatrix(inSize, outSize);
        this.hd2Wdi = new DoubleMatrix(inTmSize, outSize);
        this.hd2Whi = new DoubleMatrix(outSize, outSize);
        this.hd2Wci = new DoubleMatrix(outSize, outSize);
        this.hd2bi = new DoubleMatrix(1, outSize);
        
        this.hdWxf = new DoubleMatrix(inSize, outSize);
        this.hdWdf = new DoubleMatrix(inTmSize, outSize);
        this.hdWhf = new DoubleMatrix(outSize, outSize);
        this.hdWcf = new DoubleMatrix(outSize, outSize);
        this.hdbf = new DoubleMatrix(1, outSize);
        
        this.hd2Wxf = new DoubleMatrix(inSize, outSize);
        this.hd2Wdf = new DoubleMatrix(inTmSize, outSize);
        this.hd2Whf = new DoubleMatrix(outSize, outSize);
        this.hd2Wcf = new DoubleMatrix(outSize, outSize);
        this.hd2bf = new DoubleMatrix(1, outSize);
        
        this.hdWxc = new DoubleMatrix(inSize, outSize);
        this.hdWdc = new DoubleMatrix(inTmSize, outSize);
        this.hdWhc = new DoubleMatrix(outSize, outSize);
        this.hdbc = new DoubleMatrix(1, outSize);
        
        this.hd2Wxc = new DoubleMatrix(inSize, outSize);
        this.hd2Wdc = new DoubleMatrix(inTmSize, outSize);
        this.hd2Whc = new DoubleMatrix(outSize, outSize);
        this.hd2bc = new DoubleMatrix(1, outSize);
        
        this.hdWxo = new DoubleMatrix(inSize, outSize);
        this.hdWdo = new DoubleMatrix(inTmSize, outSize);
        this.hdWho = new DoubleMatrix(outSize, outSize);
        this.hdWco = new DoubleMatrix(outSize, outSize);
        this.hdbo = new DoubleMatrix(1, outSize);
        
        this.hd2Wxo = new DoubleMatrix(inSize, outSize);
        this.hd2Wdo = new DoubleMatrix(inTmSize, outSize);
        this.hd2Who = new DoubleMatrix(outSize, outSize);
        this.hd2Wco = new DoubleMatrix(outSize, outSize);
        this.hd2bo = new DoubleMatrix(1, outSize);
        
        this.hdWhd = new DoubleMatrix(outSize, nodeSize);
        this.hdbd = new DoubleMatrix(1, nodeSize);
        
        this.hd2Whd = new DoubleMatrix(outSize, nodeSize);
        this.hd2bd = new DoubleMatrix(1, nodeSize);
        
        this.hdWhy = new DoubleMatrix(outSize, nodeSize);
        this.hdby = new DoubleMatrix(1, nodeSize);
        
        this.hd2Why = new DoubleMatrix(outSize, nodeSize);
        this.hd2by = new DoubleMatrix(1, nodeSize);
    }
    
    public void active(int t, InputNeuron input, int node, Map<String, DoubleMatrix> acts) {
        DoubleMatrix x = input.repMatrix.getRow(node);
        acts.put("x" + t, x);
        DoubleMatrix tmFeat = input.tmFeat;
        acts.put("tmFeat" + t, tmFeat);
        DoubleMatrix preH = null, preC = null;
        if (t == 0) {
            preH = new DoubleMatrix(1, outSize);
            preC = preH.dup();
        } else {
            preH = acts.get("h" + (t - 1));
            preC = acts.get("c" + (t - 1));
        }
        
        DoubleMatrix i = Activer.logistic(x.mmul(Wxi).add(tmFeat.mmul(Wdi)).add(preH.mmul(Whi)).add(preC.mmul(Wci)).add(bi));
        DoubleMatrix f = Activer.logistic(x.mmul(Wxf).add(tmFeat.mmul(Wdf)).add(preH.mmul(Whf)).add(preC.mmul(Wcf)).add(bf));
        DoubleMatrix gc = Activer.tanh(x.mmul(Wxc).add(tmFeat.mmul(Wdc)).add(preH.mmul(Whc)).add(bc));
        DoubleMatrix c = f.mul(preC).add(i.mul(gc));
        DoubleMatrix o = Activer.logistic(x.mmul(Wxo).add(tmFeat.mmul(Wdo)).add(preH.mmul(Who)).add(c.mmul(Wco)).add(bo));
        DoubleMatrix gh = Activer.tanh(c);
        DoubleMatrix h = o.mul(gh);
        
        acts.put("i" + t, i);
        acts.put("f" + t, f);
        acts.put("gc" + t, gc);
        acts.put("c" + t, c);
        acts.put("o" + t, o);
        acts.put("gh" + t, gh);
        acts.put("h" + t, h);
    }
    
    /**
     * 
     * @param input
     * @param ndList must be the list for next nodes
     * @param tmList must be the list for the time gap
     * @param acts
     * @param lastT
     */
    public void bptt(InputNeuron input, List<Integer> ndList, List<Double> tmList
    					, Map<String, DoubleMatrix> acts, int lastT) {
    	
        for (int t = lastT; t > -1; t--) {
        	double tmGap = tmList.get(t);
        	// delta d
        	DoubleMatrix lambda = acts.get("lambda" + t);
        	DoubleMatrix deltaD = new DoubleMatrix(lambda.rows, lambda.columns); 
        	if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
        		deltaD = lambda.div(input.w).mul(MatrixFunctions.exp(input.w.mul(tmGap)).sub(1));
        		for(int k=0; k<lambda.length; k++) {
        			if(lambda.get(k)<0) {
        				deltaD.put(k, deltaD.get(k)-lambda.get(k)*AlgCons.gamma);
        			} else {
        				deltaD.put(k, deltaD.get(k)+lambda.get(k)*AlgCons.gamma);
        			}
        		}
        	}
        	if(AlgCons.tmDist.equalsIgnoreCase("const")) {
        		for(int k=0; k<lambda.length; k++) {
            		if(lambda.get(k)<0) {
            			deltaD.put(k, lambda.get(k)*(tmGap-AlgCons.gamma));
            		} else {
            			deltaD.put(k, lambda.get(k)*(tmGap+AlgCons.gamma));
            		}
            	}
        	}
            int ndIdx = ndList.get(t);
            deltaD.put(ndIdx, deltaD.get(ndIdx)-1);
            acts.put("dd" + t, deltaD);
            // delta y
            DoubleMatrix py = acts.get("py" + t);
            DoubleMatrix y = acts.get("y" + t);
            DoubleMatrix deltaY = py.sub(y);
            acts.put("dy" + t, deltaY);
            
            // cell output errors
            DoubleMatrix h = acts.get("h" + t);
            DoubleMatrix deltaH = null;
            if (t == lastT) {
            	deltaH = deltaD.mmul(Whd.transpose())
                		.add(deltaY.mmul(Why.transpose()));
            } else {
                DoubleMatrix lateDgc = acts.get("dgc" + (t + 1));
                DoubleMatrix lateDf = acts.get("df" + (t + 1));
                DoubleMatrix lateDo = acts.get("do" + (t + 1));
                DoubleMatrix lateDi = acts.get("di" + (t + 1));
                deltaH = deltaD.mmul(Whd.transpose())
                		.add(deltaY.mmul(Why.transpose()))
                        .add(lateDgc.mmul(Whc.transpose()))
                        .add(lateDi.mmul(Whi.transpose()))
                        .add(lateDo.mmul(Who.transpose()))
                        .add(lateDf.mmul(Whf.transpose()));
            }
            acts.put("dh" + t, deltaH);
            
            
            // output gates
            DoubleMatrix gh = acts.get("gh" + t);
            DoubleMatrix o = acts.get("o" + t);
            DoubleMatrix deltaO = deltaH.mul(gh).mul(deriveExp(o));
            acts.put("do" + t, deltaO);
            
            // status
            DoubleMatrix deltaC = null;
            if (t == lastT) {
                deltaC = deltaH.mul(o).mul(deriveTanh(gh))
                        .add(deltaO.mmul(Wco.transpose()));
            } else {
                DoubleMatrix lateDc = acts.get("dc" + (t + 1));
                DoubleMatrix lateDf = acts.get("df" + (t + 1));
                DoubleMatrix lateF = acts.get("f" + (t + 1));
                DoubleMatrix lateDi = acts.get("di" + (t + 1));
                deltaC = deltaH.mul(o).mul(deriveTanh(gh)) 
                        .add(deltaO.mmul(Wco.transpose())) //output gate related
                        .add(lateF.mul(lateDc)) //cells related
                        .add(lateDf.mmul(Wcf.transpose())) //forget gate related
                        .add(lateDi.mmul(Wci.transpose())); //input gate related
            }
            acts.put("dc" + t, deltaC);
            
            // cells
            DoubleMatrix gc = acts.get("gc" + t);
            DoubleMatrix i = acts.get("i" + t);
            DoubleMatrix deltaGc = deltaC.mul(i).mul(deriveTanh(gc));
            acts.put("dgc" + t, deltaGc);
        
            DoubleMatrix preC = null;
            if (t > 0) {
                preC = acts.get("c" + (t - 1));
            } else {
                preC = DoubleMatrix.zeros(1, h.length);
            }
            // forget gates
            DoubleMatrix f = acts.get("f" + t);
            DoubleMatrix deltaF = deltaC.mul(preC).mul(deriveExp(f));
            acts.put("df" + t, deltaF);
        
            // input gates
            DoubleMatrix deltaI = deltaC.mul(gc).mul(deriveExp(i));
            acts.put("di" + t, deltaI);
        }
        calcWeightsGradient(input, acts, lastT);
    }
    
    private void calcWeightsGradient(InputNeuron input, Map<String, DoubleMatrix> acts, int lastT) {
    	
        DoubleMatrix dWxi = new DoubleMatrix(Wxi.rows, Wxi.columns);
        DoubleMatrix dWdi = new DoubleMatrix(Wdi.rows, Wdi.columns);
        DoubleMatrix dWhi = new DoubleMatrix(Whi.rows, Whi.columns);
        DoubleMatrix dWci = new DoubleMatrix(Wci.rows, Wci.columns);
        DoubleMatrix dbi = new DoubleMatrix(bi.rows, bi.columns);
        
        DoubleMatrix dWxf = new DoubleMatrix(Wxf.rows, Wxf.columns);
        DoubleMatrix dWdf = new DoubleMatrix(Wdf.rows, Wdf.columns);
        DoubleMatrix dWhf = new DoubleMatrix(Whf.rows, Whf.columns);
        DoubleMatrix dWcf = new DoubleMatrix(Wcf.rows, Wcf.columns);
        DoubleMatrix dbf = new DoubleMatrix(bf.rows, bf.columns);
        
        DoubleMatrix dWxc = new DoubleMatrix(Wxc.rows, Wxc.columns);
        DoubleMatrix dWdc = new DoubleMatrix(Wdc.rows, Wdc.columns);
        DoubleMatrix dWhc = new DoubleMatrix(Whc.rows, Whc.columns);
        DoubleMatrix dbc = new DoubleMatrix(bc.rows, bc.columns);
        
        DoubleMatrix dWxo = new DoubleMatrix(Wxo.rows, Wxo.columns);
        DoubleMatrix dWdo = new DoubleMatrix(Wdo.rows, Wdo.columns);
        DoubleMatrix dWho = new DoubleMatrix(Who.rows, Who.columns);
        DoubleMatrix dWco = new DoubleMatrix(Wco.rows, Wco.columns);
        DoubleMatrix dbo = new DoubleMatrix(bo.rows, bo.columns);
        
        DoubleMatrix dWhd = new DoubleMatrix(Whd.rows, Whd.columns);
        DoubleMatrix dbd = new DoubleMatrix(bd.rows, bd.columns);
        
        DoubleMatrix dWhy = new DoubleMatrix(Why.rows, Why.columns);
        DoubleMatrix dby = new DoubleMatrix(by.rows, by.columns);
        
        for (int t = 0; t < lastT + 1; t++) {
            DoubleMatrix x = acts.get("x" + t).transpose();
            DoubleMatrix tmFeat = acts.get("tmFeat" + t).transpose();
            
            dWxi = dWxi.add(x.mmul(acts.get("di" + t)));
            dWxf = dWxf.add(x.mmul(acts.get("df" + t)));
            dWxc = dWxc.add(x.mmul(acts.get("dgc" + t)));
            dWxo = dWxo.add(x.mmul(acts.get("do" + t)));
            
            dWdi = dWdi.add(tmFeat.mmul(acts.get("di" + t)));
            dWdf = dWdf.add(tmFeat.mmul(acts.get("df" + t)));
            dWdc = dWdc.add(tmFeat.mmul(acts.get("dgc" + t)));
            dWdo = dWdo.add(tmFeat.mmul(acts.get("do" + t)));
            
            if (t > 0) {
                DoubleMatrix preH = acts.get("h" + (t - 1)).transpose();
                DoubleMatrix preC = acts.get("c" + (t - 1)).transpose();
                dWhi = dWhi.add(preH.mmul(acts.get("di" + t)));
                dWhf = dWhf.add(preH.mmul(acts.get("df" + t)));
                dWhc = dWhc.add(preH.mmul(acts.get("dgc" + t)));
                dWho = dWho.add(preH.mmul(acts.get("do" + t)));
                dWci = dWci.add(preC.mmul(acts.get("di" + t)));
                dWcf = dWcf.add(preC.mmul(acts.get("df" + t)));
            }
            dWco = dWco.add(acts.get("c" + t).transpose().mmul(acts.get("do" + t)));
            dWhd = dWhd.add(acts.get("h" + t).transpose().mmul(acts.get("dd" + t)));
            dWhy = dWhy.add(acts.get("h" + t).transpose().mmul(acts.get("dy" + t)));
            
            dbi = dbi.add(acts.get("di" + t));
            dbf = dbf.add(acts.get("df" + t));
            dbc = dbc.add(acts.get("dgc" + t));
            dbo = dbo.add(acts.get("do" + t));
            dbd = dbd.add(acts.get("dd" + t));
            dby = dby.add(acts.get("dy" + t));
        }
        
        acts.put("dWxi", dWxi);
        acts.put("dWdi", dWdi);
        acts.put("dWhi", dWhi);
        acts.put("dWci", dWci);
        acts.put("dbi", dbi);
        
        acts.put("dWxf", dWxf);
        acts.put("dWdf", dWdf);
        acts.put("dWhf", dWhf);
        acts.put("dWcf", dWcf);
        acts.put("dbf", dbf);
        
        acts.put("dWxc", dWxc);
        acts.put("dWdc", dWdc);
        acts.put("dWhc", dWhc);
        acts.put("dbc", dbc);
        
        acts.put("dWxo", dWxo);
        acts.put("dWdo", dWdo);
        acts.put("dWho", dWho);
        acts.put("dWco", dWco);
        acts.put("dbo", dbo);
        
        acts.put("dWhd", dWhd);
        acts.put("dbd", dbd);
        
        acts.put("dWhy", dWhy);
        acts.put("dby", dby);
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	LSTMBatchDerivative batchDerv = (LSTMBatchDerivative) derv;
    	
        hdWxi = hdWxi.add(MatrixFunctions.pow(batchDerv.dWxi, 2.));
        hdWdi = hdWdi.add(MatrixFunctions.pow(batchDerv.dWdi, 2.));
        hdWhi = hdWhi.add(MatrixFunctions.pow(batchDerv.dWhi, 2.));
        hdWci = hdWci.add(MatrixFunctions.pow(batchDerv.dWci, 2.));
        hdbi = hdbi.add(MatrixFunctions.pow(batchDerv.dbi, 2.));
        
        hdWxf = hdWxf.add(MatrixFunctions.pow(batchDerv.dWxf, 2.));
        hdWdf = hdWdf.add(MatrixFunctions.pow(batchDerv.dWdf, 2.));
        hdWhf = hdWhf.add(MatrixFunctions.pow(batchDerv.dWhf, 2.));
        hdWcf = hdWcf.add(MatrixFunctions.pow(batchDerv.dWcf, 2.));
        hdbf = hdbf.add(MatrixFunctions.pow(batchDerv.dbf, 2.));
        
        hdWxc = hdWxc.add(MatrixFunctions.pow(batchDerv.dWxc, 2.));
        hdWdc = hdWdc.add(MatrixFunctions.pow(batchDerv.dWdc, 2.));
        hdWhc = hdWhc.add(MatrixFunctions.pow(batchDerv.dWhc, 2.));
        hdbc = hdbc.add(MatrixFunctions.pow(batchDerv.dbc, 2.));
        
        hdWxo = hdWxo.add(MatrixFunctions.pow(batchDerv.dWxo, 2.));
        hdWdo = hdWdo.add(MatrixFunctions.pow(batchDerv.dWdo, 2.));
        hdWho = hdWho.add(MatrixFunctions.pow(batchDerv.dWho, 2.));
        hdWco = hdWco.add(MatrixFunctions.pow(batchDerv.dWco, 2.));
        hdbo = hdbo.add(MatrixFunctions.pow(batchDerv.dbo, 2.));
        
        hdWhd = hdWhd.add(MatrixFunctions.pow(batchDerv.dWhd, 2.));
        hdbd = hdbd.add(MatrixFunctions.pow(batchDerv.dbd, 2.));
        
        hdWhy = hdWhy.add(MatrixFunctions.pow(batchDerv.dWhy, 2.));
        hdby = hdby.add(MatrixFunctions.pow(batchDerv.dby, 2.));
        
        Wxi = Wxi.sub(batchDerv.dWxi.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxi).add(eps),-1.).mul(lr)));
        Wdi = Wdi.sub(batchDerv.dWdi.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdi).add(eps),-1.).mul(lr)));
        Whi = Whi.sub(batchDerv.dWhi.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhi).add(eps),-1.).mul(lr)));
        Wci = Wci.sub(batchDerv.dWci.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWci).add(eps),-1.).mul(lr)));
        bi = bi.sub(batchDerv.dbi.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbi).add(eps),-1.).mul(lr)));
        
        Wxf = Wxf.sub(batchDerv.dWxf.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxf).add(eps),-1.).mul(lr)));
        Wdf = Wdf.sub(batchDerv.dWdf.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdf).add(eps),-1.).mul(lr)));
        Whf = Whf.sub(batchDerv.dWhf.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhf).add(eps),-1.).mul(lr)));
        Wcf = Wcf.sub(batchDerv.dWcf.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWcf).add(eps),-1.).mul(lr)));
        bf = bf.sub(batchDerv.dbf.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbf).add(eps),-1.).mul(lr)));
        
        Wxc = Wxc.sub(batchDerv.dWxc.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxc).add(eps),-1.).mul(lr)));
        Wdc = Wdc.sub(batchDerv.dWdc.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdc).add(eps),-1.).mul(lr)));
        Whc = Whc.sub(batchDerv.dWhc.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhc).add(eps),-1.).mul(lr)));
        bc = bc.sub(batchDerv.dbc.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbc).add(eps),-1.).mul(lr)));

        Wxo = Wxo.sub(batchDerv.dWxo.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxo).add(eps),-1.).mul(lr)));
        Wdo = Wdo.sub(batchDerv.dWdo.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdo).add(eps),-1.).mul(lr)));
        Who = Who.sub(batchDerv.dWho.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWho).add(eps),-1.).mul(lr)));
        Wco = Wco.sub(batchDerv.dWco.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWco).add(eps),-1.).mul(lr)));
        bo = bo.sub(batchDerv.dbo.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbo).add(eps),-1.).mul(lr)));
        
        Whd = Whd.sub(batchDerv.dWhd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhd).add(eps),-1.).mul(lr)));
        bd = bd.sub(batchDerv.dbd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbd).add(eps),-1.).mul(lr)));
        
        Why = Why.sub(batchDerv.dWhy.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhy).add(eps),-1.).mul(lr)));
        by = by.sub(batchDerv.dby.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdby).add(eps),-1.).mul(lr)));
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
    						, double beta1, double beta2, int epochT) {
    	
    	LSTMBatchDerivative batchDerv = (LSTMBatchDerivative) derv;

		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		hd2Wxi = hd2Wxi.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxi, 2.).mul(1 - beta2));
		hd2Wdi = hd2Wdi.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdi, 2.).mul(1 - beta2));
		hd2Whi = hd2Whi.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhi, 2.).mul(1 - beta2));
		hd2Wci = hd2Wci.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWci, 2.).mul(1 - beta2));
		hd2bi = hd2bi.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbi, 2.).mul(1 - beta2));
		
		hdWxi = hdWxi.mul(beta1).add(batchDerv.dWxi.mul(1 - beta1));
		hdWdi = hdWdi.mul(beta1).add(batchDerv.dWdi.mul(1 - beta1));
		hdWhi = hdWhi.mul(beta1).add(batchDerv.dWhi.mul(1 - beta1));
		hdWci = hdWci.mul(beta1).add(batchDerv.dWci.mul(1 - beta1));
		hdbi = hdbi.mul(beta1).add(batchDerv.dbi.mul(1 - beta1));

		hd2Wxf = hd2Wxf.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxf, 2.).mul(1 - beta2));
		hd2Wdf = hd2Wdf.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdf, 2.).mul(1 - beta2));
		hd2Whf = hd2Whf.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhf, 2.).mul(1 - beta2));
		hd2Wcf = hd2Wcf.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWcf, 2.).mul(1 - beta2));
		hd2bf = hd2bf.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbf, 2.).mul(1 - beta2));
		
		hdWxf = hdWxf.mul(beta1).add(batchDerv.dWxf.mul(1 - beta1));
		hdWdf = hdWdf.mul(beta1).add(batchDerv.dWdf.mul(1 - beta1));
		hdWhf = hdWhf.mul(beta1).add(batchDerv.dWhf.mul(1 - beta1));
		hdWcf = hdWcf.mul(beta1).add(batchDerv.dWcf.mul(1 - beta1));
		hdbf = hdbf.mul(beta1).add(batchDerv.dbf.mul(1 - beta1));

		hd2Wxc = hd2Wxc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxc, 2.).mul(1 - beta2));
		hd2Wdc = hd2Wdc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdc, 2.).mul(1 - beta2));
		hd2Whc = hd2Whc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhc, 2.).mul(1 - beta2));
		hd2bc = hd2bc.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbc, 2.).mul(1 - beta2));
		
		hdWxc = hdWxc.mul(beta1).add(batchDerv.dWxc.mul(1 - beta1));
		hdWdc = hdWdc.mul(beta1).add(batchDerv.dWdc.mul(1 - beta1));
		hdWhc = hdWhc.mul(beta1).add(batchDerv.dWhc.mul(1 - beta1));
		hdbc = hdbc.mul(beta1).add(batchDerv.dbc.mul(1 - beta1));

		hd2Wxo = hd2Wxo.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxo, 2.).mul(1 - beta2));
		hd2Wdo = hd2Wdo.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdo, 2.).mul(1 - beta2));
		hd2Who = hd2Who.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWho, 2.).mul(1 - beta2));
		hd2Wco = hd2Wco.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWco, 2.).mul(1 - beta2));
		hd2bo = hd2bo.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbo, 2.).mul(1 - beta2));
		
		hdWxo = hdWxo.mul(beta1).add(batchDerv.dWxo.mul(1 - beta1));
		hdWdo = hdWdo.mul(beta1).add(batchDerv.dWdo.mul(1 - beta1));
		hdWho = hdWho.mul(beta1).add(batchDerv.dWho.mul(1 - beta1));
		hdWco = hdWco.mul(beta1).add(batchDerv.dWco.mul(1 - beta1));
		hdbo = hdbo.mul(beta1).add(batchDerv.dbo.mul(1 - beta1));

		hd2Whd = hd2Whd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhd, 2.).mul(1 - beta2));
		hd2bd = hd2bd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbd, 2.).mul(1 - beta2));
		
		hdWhd = hdWhd.mul(beta1).add(batchDerv.dWhd.mul(1 - beta1));
		hdbd = hdbd.mul(beta1).add(batchDerv.dbd.mul(1 - beta1));
		
		hd2Why = hd2Why.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhy, 2.).mul(1 - beta2));
		hd2by = hd2by.mul(beta2).add(MatrixFunctions.pow(batchDerv.dby, 2.).mul(1 - beta2));
		
		hdWhy = hdWhy.mul(beta1).add(batchDerv.dWhy.mul(1 - beta1));
		hdby = hdby.mul(beta1).add(batchDerv.dby.mul(1 - beta1));

		Wxi = Wxi.sub(
					hdWxi.mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxi.mul(biasBeta2)).add(eps), -1))
					);
		Wdi = Wdi.sub(
				hdWdi.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdi.mul(biasBeta2)).add(eps), -1))
				);
		Whi = Whi.sub(
				hdWhi.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whi.mul(biasBeta2)).add(eps), -1))
				);
		Wci = Wci.sub(
				hdWci.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wci.mul(biasBeta2)).add(eps), -1.))
				);
		bi = bi.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bi.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbi.mul(biasBeta1)).mul(lr)
				);
		
		Wxf = Wxf.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxf.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWxf.mul(biasBeta1)).mul(lr)
				);
		Wdf = Wdf.sub(
				hdWdf.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdf.mul(biasBeta2)).add(eps), -1))
				);
		Whf = Whf.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whf.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWhf.mul(biasBeta1)).mul(lr)
				);
		Wcf = Wcf.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wcf.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWcf.mul(biasBeta1)).mul(lr)
				);
		bf = bf.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bf.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbf.mul(biasBeta1)).mul(lr)
				);

		Wxc = Wxc.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxc.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWxc.mul(biasBeta1)).mul(lr)
				);
		Wdc = Wdc.sub(
				hdWdc.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdc.mul(biasBeta2)).add(eps), -1))
				);
		Whc = Whc.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whc.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWhc.mul(biasBeta1)).mul(lr)
				);
		bc = bc.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bc.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbc.mul(biasBeta1)).mul(lr)
				);

		Wxo = Wxo.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxo.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWxo.mul(biasBeta1)).mul(lr)
				);
		Wdo = Wdo.sub(
				hdWdo.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdo.mul(biasBeta2)).add(eps), -1))
				);
		Who = Who.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Who.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWho.mul(biasBeta1)).mul(lr)
				);
		Wco = Wco.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wco.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWco.mul(biasBeta1)).mul(lr)
				);
		bo = bo.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bo.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbo.mul(biasBeta1)).mul(lr)
				);
		
		Whd = Whd.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whd.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWhd.mul(biasBeta1)).mul(lr)
				);
		bd = bd.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bd.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbd.mul(biasBeta1)).mul(lr)
				);
		
		Why = Why.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Why.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWhy.mul(biasBeta1)).mul(lr)
				);
		by = by.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2by.mul(biasBeta2)).add(eps), -1.)
				.mul(hdby.mul(biasBeta1)).mul(lr)
				);
    }
    
    public DoubleMatrix dDecode(DoubleMatrix ht) {
    	return ht.mmul(Whd).add(bd);
    }
    
    /* (non-Javadoc)
	 * @see com.kingwang.ctsrnn.rnn.RNNCell#yDecode(org.jblas.DoubleMatrix)
	 */
	@Override
	public DoubleMatrix yDecode(DoubleMatrix ht) {
		return ht.mmul(Why).add(by);
	}

	public void writeRes(String outFile) {
    	
    	OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile);
    	FileUtil.writeln(osw, "Wxi");
    	writeMatrix(osw, Wxi);
    	FileUtil.writeln(osw, "Wdi");
    	writeMatrix(osw, Wdi);
    	FileUtil.writeln(osw, "Whi");
    	writeMatrix(osw, Whi);
    	FileUtil.writeln(osw, "Wci");
    	writeMatrix(osw, Wci);
    	FileUtil.writeln(osw, "bi");
    	writeMatrix(osw, bi);
    	
    	FileUtil.writeln(osw, "Wxf");
    	writeMatrix(osw, Wxf);
    	FileUtil.writeln(osw, "Wdf");
    	writeMatrix(osw, Wdf);
    	FileUtil.writeln(osw, "Whf");
    	writeMatrix(osw, Whf);
    	FileUtil.writeln(osw, "Wcf");
    	writeMatrix(osw, Wcf);
    	FileUtil.writeln(osw, "bf");
    	writeMatrix(osw, bf);
    	
    	FileUtil.writeln(osw, "Wxc");
    	writeMatrix(osw, Wxc);
    	FileUtil.writeln(osw, "Wdc");
    	writeMatrix(osw, Wdc);
    	FileUtil.writeln(osw, "Whc");
    	writeMatrix(osw, Whc);
    	FileUtil.writeln(osw, "bc");
    	writeMatrix(osw, bc);
    	
    	FileUtil.writeln(osw, "Wxo");
    	writeMatrix(osw, Wxo);
    	FileUtil.writeln(osw, "Wdo");
    	writeMatrix(osw, Wdo);
    	FileUtil.writeln(osw, "Who");
    	writeMatrix(osw, Who);
    	FileUtil.writeln(osw, "Wco");
    	writeMatrix(osw, Wco);
    	FileUtil.writeln(osw, "bo");
    	writeMatrix(osw, bo);
    	
    	FileUtil.writeln(osw, "Whd");
    	writeMatrix(osw, Whd);
    	FileUtil.writeln(osw, "bd");
    	writeMatrix(osw, bd);
    	
    	FileUtil.writeln(osw, "Why");
    	writeMatrix(osw, Why);
    	FileUtil.writeln(osw, "by");
    	writeMatrix(osw, by);
    }
    
    public void loadRNNModel(String rnnModelFile) {

    	LoadTypes type = LoadTypes.Null;
    	int row = 0;
    	
    	try(BufferedReader br = FileUtil.getBufferReader(rnnModelFile)) {
    		String line = null;
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(",");
    			if(elems.length<2 && !elems[0].contains(".")) {
    				type = LoadTypes.valueOf(elems[0]);
    				row = 0;
    				continue;
    			}
    			switch(type) {
	    			case Wxi: this.Wxi = matrixSetter(row, elems, this.Wxi); break;
	    			case Wdi: this.Wdi = matrixSetter(row, elems, this.Wdi); break;
	    			case Whi: this.Whi = matrixSetter(row, elems, this.Whi); break;
	    			case Wci: this.Wci = matrixSetter(row, elems, this.Wci); break;
	    			case bi: this.bi = matrixSetter(row, elems, this.bi); break;
	    			
	    			case Wxf: this.Wxf = matrixSetter(row, elems, this.Wxf); break;
	    			case Wdf: this.Wdf = matrixSetter(row, elems, this.Wdf); break;
	    			case Whf: this.Whf = matrixSetter(row, elems, this.Whf); break;
	    			case Wcf: this.Wcf = matrixSetter(row, elems, this.Wcf); break;
	    			case bf: this.bf = matrixSetter(row, elems, this.bf); break;
	    			
	    			case Wxc: this.Wxc = matrixSetter(row, elems, this.Wxc); break;
	    			case Wdc: this.Wdc = matrixSetter(row, elems, this.Wdc); break;
	    			case Whc: this.Whc = matrixSetter(row, elems, this.Whc); break;
	    			case bc: this.bc = matrixSetter(row, elems, this.bc); break;
	    			
	    			case Wxo: this.Wxo = matrixSetter(row, elems, this.Wxo); break;
	    			case Wdo: this.Wdo = matrixSetter(row, elems, this.Wdo); break;
	    			case Who: this.Who = matrixSetter(row, elems, this.Who); break;
	    			case Wco: this.Wco = matrixSetter(row, elems, this.Wco); break;
	    			case bo: this.bo = matrixSetter(row, elems, this.bo); break;
	    			
	    			case Whd: this.Whd = matrixSetter(row, elems, this.Whd); break;
	    			case bd: this.bd = matrixSetter(row, elems, this.bd); break;
	    			
	    			case Why: this.Why = matrixSetter(row, elems, this.Why); break;
	    			case by: this.by = matrixSetter(row, elems, this.by); break;
    			}
    			row++;
    		}
    		
    	} catch(IOException e) {
    		
    	}
    }

}