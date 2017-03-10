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
import com.kingwang.rnntd.batchderv.impl.GRUBatchDerivative;
import com.kingwang.rnntd.cell.Operator;
import com.kingwang.rnntd.cell.RNNCell;
import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.utils.Activer;
import com.kingwang.rnntd.utils.LoadTypes;
import com.kingwang.rnntd.utils.MatIniter;
import com.kingwang.rnntd.utils.MatIniter.Type;

public class GRU extends Operator implements RNNCell, Serializable{
	
    private static final long serialVersionUID = -1501734916541393551L;

    private int outSize;
    
    /**
     * historical first-derivative gradient of weights 
     */
    public DoubleMatrix hdWxr;
    public DoubleMatrix hdWdr;
    public DoubleMatrix hdWhr;
    public DoubleMatrix hdbr;
    
    public DoubleMatrix hdWxz;
    public DoubleMatrix hdWdz;
    public DoubleMatrix hdWhz;
    public DoubleMatrix hdbz;
    
    public DoubleMatrix hdWxh;
    public DoubleMatrix hdWdh;
    public DoubleMatrix hdWhh;
    public DoubleMatrix hdbh;
    
    public DoubleMatrix hdWhd;
    public DoubleMatrix hdbd;
    
    public DoubleMatrix hdWhy;
    public DoubleMatrix hdby;
    
    /**
     * historical second-derivative gradient of weights 
     */
    public DoubleMatrix hd2Wxr;
    public DoubleMatrix hd2Wdr;
    public DoubleMatrix hd2Whr;
    public DoubleMatrix hd2br;
    
    public DoubleMatrix hd2Wxz;
    public DoubleMatrix hd2Wdz;
    public DoubleMatrix hd2Whz;
    public DoubleMatrix hd2bz;
    
    public DoubleMatrix hd2Wxh;
    public DoubleMatrix hd2Wdh;
    public DoubleMatrix hd2Whh;
    public DoubleMatrix hd2bh;
    
    public DoubleMatrix hd2Whd;
    public DoubleMatrix hd2bd;
    
    public DoubleMatrix hd2Why;
    public DoubleMatrix hd2by;
    
    /**
     * model parameters
     */
    public DoubleMatrix Wxr;
    public DoubleMatrix Wdr;
    public DoubleMatrix Whr;
    public DoubleMatrix br;
    
    public DoubleMatrix Wxz;
    public DoubleMatrix Wdz;
    public DoubleMatrix Whz;
    public DoubleMatrix bz;
    
    public DoubleMatrix Wxh;
    public DoubleMatrix Wdh;
    public DoubleMatrix Whh;
    public DoubleMatrix bh;
    
    public DoubleMatrix Whd;
    public DoubleMatrix bd;
    
    public DoubleMatrix Why;
    public DoubleMatrix by;
    
    public GRU(int inSize, int inTmSize, int outSize, int nodeSize, MatIniter initer) {
        this.outSize = outSize;
        
        if (initer.getType() == Type.Uniform) {
            this.Wxr = initer.uniform(inSize, outSize);
            this.Wdr = initer.uniform(inTmSize, outSize);
            this.Whr = initer.uniform(outSize, outSize);
            this.br = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxz = initer.uniform(inSize, outSize);
            this.Wdz = initer.uniform(inTmSize, outSize);
            this.Whz = initer.uniform(outSize, outSize);
            this.bz = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxh = initer.uniform(inSize, outSize);
            this.Wdh = initer.uniform(inTmSize, outSize);
            this.Whh = initer.uniform(outSize, outSize);
            this.bh = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Whd = initer.uniform(outSize, nodeSize);
            this.bd = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
            
            this.Why = initer.uniform(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
        	this.Wxr = initer.gaussian(inSize, outSize);
            this.Wdr = initer.gaussian(inTmSize, outSize);
            this.Whr = initer.gaussian(outSize, outSize);
            this.br = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxz = initer.gaussian(inSize, outSize);
            this.Wdz = initer.gaussian(inTmSize, outSize);
            this.Whz = initer.gaussian(outSize, outSize);
            this.bz = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxh = initer.gaussian(inSize, outSize);
            this.Wdh = initer.gaussian(inTmSize, outSize);
            this.Whh = initer.gaussian(outSize, outSize);
            this.bh = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Whd = initer.gaussian(outSize, nodeSize);
            this.bd = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
            
            this.Why = initer.gaussian(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
        	this.Wxr = initer.svd(inSize, outSize);
            this.Wdr = initer.svd(inTmSize, outSize);
            this.Whr = initer.svd(outSize, outSize);
            this.br = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxz = initer.svd(inSize, outSize);
            this.Wdz = initer.svd(inTmSize, outSize);
            this.Whz = initer.svd(outSize, outSize);
            this.bz = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Wxh = initer.svd(inSize, outSize);
            this.Wdh = initer.svd(inTmSize, outSize);
            this.Whh = initer.svd(outSize, outSize);
            this.bh = new DoubleMatrix(1, outSize).add(AlgCons.biasInitVal);
            
            this.Whd = initer.svd(outSize, nodeSize);
            this.bd = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
            
            this.Why = initer.svd(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        	this.Wxr = DoubleMatrix.zeros(inSize, outSize).add(0.1);
        	this.Wdr = DoubleMatrix.zeros(inTmSize, outSize).add(0.1);
            this.Whr = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.br = DoubleMatrix.zeros(1, outSize).add(0.4);
            
            this.Wxz = DoubleMatrix.zeros(inSize, outSize).add(0.1);
            this.Wdz = DoubleMatrix.zeros(inTmSize, outSize).add(0.1);
            this.Whz = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.bz = DoubleMatrix.zeros(1, outSize).add(0.4);
            
            this.Wxh = DoubleMatrix.zeros(inSize, outSize).add(0.1);
            this.Wdh = DoubleMatrix.zeros(inTmSize, outSize).add(0.1);
            this.Whh = DoubleMatrix.zeros(outSize, outSize).add(0.2);
            this.bh = DoubleMatrix.zeros(1, outSize).add(0.3);
            
            this.Whd = DoubleMatrix.zeros(outSize, nodeSize).add(.1);
            this.bd = new DoubleMatrix(1, nodeSize).add(.2);
        }
        
        this.hdWxr = new DoubleMatrix(inSize, outSize);
        this.hdWdr = new DoubleMatrix(inTmSize, outSize);
        this.hdWhr = new DoubleMatrix(outSize, outSize);
        this.hdbr = new DoubleMatrix(1, outSize);
        
        this.hdWxz = new DoubleMatrix(inSize, outSize);
        this.hdWdz = new DoubleMatrix(inTmSize, outSize);
        this.hdWhz = new DoubleMatrix(outSize, outSize);
        this.hdbz = new DoubleMatrix(1, outSize);
        
        this.hdWxh = new DoubleMatrix(inSize, outSize);
        this.hdWdh = new DoubleMatrix(inTmSize, outSize);
        this.hdWhh = new DoubleMatrix(outSize, outSize);
        this.hdbh = new DoubleMatrix(1, outSize);
        
        this.hdWhd = new DoubleMatrix(outSize, nodeSize);
        this.hdbd = new DoubleMatrix(1, nodeSize);
        
        this.hdWhy = new DoubleMatrix(outSize, nodeSize);
        this.hdby = new DoubleMatrix(1, nodeSize);
        
        this.hd2Wxr = new DoubleMatrix(inSize, outSize);
        this.hd2Wdr = new DoubleMatrix(inTmSize, outSize);
        this.hd2Whr = new DoubleMatrix(outSize, outSize);
        this.hd2br = new DoubleMatrix(1, outSize);
        
        this.hd2Wxz = new DoubleMatrix(inSize, outSize);
        this.hd2Wdz = new DoubleMatrix(inTmSize, outSize);
        this.hd2Whz = new DoubleMatrix(outSize, outSize);
        this.hd2bz = new DoubleMatrix(1, outSize);
        
        this.hd2Wxh = new DoubleMatrix(inSize, outSize);
        this.hd2Wdh = new DoubleMatrix(inTmSize, outSize);
        this.hd2Whh = new DoubleMatrix(outSize, outSize);
        this.hd2bh = new DoubleMatrix(1, outSize);
        
        this.hd2Whd = new DoubleMatrix(outSize, nodeSize);
        this.hd2bd = new DoubleMatrix(1, nodeSize);
        
        this.hd2Why = new DoubleMatrix(outSize, nodeSize);
        this.hd2by = new DoubleMatrix(1, nodeSize);
    }
    
    public void active(int t, InputNeuron input, int node, Map<String, DoubleMatrix> acts) {
        DoubleMatrix x = input.repMatrix.getRow(node);
        acts.put("x" + t, x);
        DoubleMatrix tmFeat = input.tmFeat;
        acts.put("tmFeat" + t, tmFeat);
        
        DoubleMatrix preH = null;
        if (t == 0) {
            preH = new DoubleMatrix(1, outSize);
        } else {
            preH = acts.get("h" + (t - 1));
        }
        
        DoubleMatrix r = Activer.logistic(x.mmul(Wxr).add(tmFeat.mmul(Wdr)).add(preH.mmul(Whr)).add(br));
        DoubleMatrix z = Activer.logistic(x.mmul(Wxz).add(tmFeat.mmul(Wdz)).add(preH.mmul(Whz)).add(bz));
        DoubleMatrix gh = Activer.tanh(x.mmul(Wxh).add(tmFeat.mmul(Wdh)).add(r.mul(preH.mmul(Whh))).add(bh));
        DoubleMatrix h = z.mul(preH).add((DoubleMatrix.ones(1, z.columns).sub(z)).mul(gh));
        
        acts.put("r" + t, r);
        acts.put("z" + t, z);
        acts.put("gh" + t, gh);
        acts.put("h" + t, h);
    }
    
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
            DoubleMatrix z = acts.get("z" + t);
            DoubleMatrix r = acts.get("r" + t);
            DoubleMatrix gh = acts.get("gh" + t);
            
            DoubleMatrix deltaH = null;
            if (t == lastT) {
            	deltaH = deltaD.mmul(Whd.transpose())
            			.add(deltaY.mmul(Why.transpose()));
            } else {
                DoubleMatrix lateDh = acts.get("dh" + (t + 1));
                DoubleMatrix lateDgh = acts.get("dgh" + (t + 1));
                DoubleMatrix lateDr = acts.get("dr" + (t + 1));
                DoubleMatrix lateDz = acts.get("dz" + (t + 1));
                DoubleMatrix lateR = acts.get("r" + (t + 1));
                DoubleMatrix lateZ = acts.get("z" + (t + 1));
                deltaH = deltaD.mmul(Whd.transpose())
                		.add(deltaY.mmul(Why.transpose()))
                        .add(lateDr.mmul(Whr.transpose()))
                        .add(lateDz.mmul(Whz.transpose()))
                        .add(lateDgh.mul(lateR).mmul(Whh.transpose()))
                        .add(lateDh.mul(lateZ));
            }
            acts.put("dh" + t, deltaH);
            
            // gh
            DoubleMatrix deltaGh = deltaH.mul(DoubleMatrix.ones(1, z.columns).sub(z)).mul(deriveTanh(gh));
            acts.put("dgh" + t, deltaGh);
            
            DoubleMatrix preH = null;
            if (t > 0) {
                preH = acts.get("h" + (t - 1));
            } else {
                preH = DoubleMatrix.zeros(1, h.length);
            }
            
            // reset gates
            DoubleMatrix deltaR = preH.mmul(Whh).mul(deltaGh).mul(deriveExp(r));
            acts.put("dr" + t, deltaR);
            
            // update gates
            DoubleMatrix deltaZ = deltaH.mul(preH.sub(gh)).mul(deriveExp(z));
            acts.put("dz" + t, deltaZ);
        }
        calcWeightsGradient(input, acts, lastT);
    }
    
    private void calcWeightsGradient(InputNeuron input, Map<String, DoubleMatrix> acts, int lastT) {
        DoubleMatrix dWxr = new DoubleMatrix(Wxr.rows, Wxr.columns);
        DoubleMatrix dWdr = new DoubleMatrix(Wdr.rows, Wdr.columns);
        DoubleMatrix dWhr = new DoubleMatrix(Whr.rows, Whr.columns);
        DoubleMatrix dbr = new DoubleMatrix(br.rows, br.columns);
        
        DoubleMatrix dWxz = new DoubleMatrix(Wxz.rows, Wxz.columns);
        DoubleMatrix dWdz = new DoubleMatrix(Wdz.rows, Wdz.columns);
        DoubleMatrix dWhz = new DoubleMatrix(Whz.rows, Whz.columns);
        DoubleMatrix dbz = new DoubleMatrix(bz.rows, bz.columns);
        
        DoubleMatrix dWxh = new DoubleMatrix(Wxh.rows, Wxh.columns);
        DoubleMatrix dWdh = new DoubleMatrix(Wdh.rows, Wdh.columns);
        DoubleMatrix dWhh = new DoubleMatrix(Whh.rows, Whh.columns);
        DoubleMatrix dbh = new DoubleMatrix(bh.rows, bh.columns);
        
        DoubleMatrix dWhd = new DoubleMatrix(Whd.rows, Whd.columns);
        DoubleMatrix dbd = new DoubleMatrix(bd.rows, bd.columns);
        
        DoubleMatrix dWhy = new DoubleMatrix(Why.rows, Why.columns);
        DoubleMatrix dby = new DoubleMatrix(by.rows, by.columns);
        
        for (int t = 0; t < lastT + 1; t++) {
        	DoubleMatrix x = acts.get("x" + t).transpose();
            DoubleMatrix tmFeat = acts.get("tmFeat" + t).transpose();
            
            dWxr = dWxr.add(x.mmul(acts.get("dr" + t)));
            dWxz = dWxz.add(x.mmul(acts.get("dz" + t)));
            dWxh = dWxh.add(x.mmul(acts.get("dgh" + t)));
            
            dWdr = dWdr.add(tmFeat.mmul(acts.get("dr" + t)));
            dWdz = dWdz.add(tmFeat.mmul(acts.get("dz" + t)));
            dWdh = dWdh.add(tmFeat.mmul(acts.get("dgh" + t)));
            
            if (t > 0) {
                DoubleMatrix preH = acts.get("h" + (t - 1)).transpose();
                dWhr = dWhr.add(preH.mmul(acts.get("dr" + t)));
                dWhz = dWhz.add(preH.mmul(acts.get("dz" + t)));
                dWhh = dWhh.add(preH.mmul(acts.get("r" + t).mul(acts.get("dgh" + t))));
            }
            dWhd = dWhd.add(acts.get("h" + t).transpose().mmul(acts.get("dd" + t)));
            dWhy = dWhy.add(acts.get("h" + t).transpose().mmul(acts.get("dy" + t)));
            
            dbr = dbr.add(acts.get("dr" + t));
            dbz = dbz.add(acts.get("dz" + t));
            dbh = dbh.add(acts.get("dgh" + t));
            dbd = dbd.add(acts.get("dd" + t));
            dby = dby.add(acts.get("dy" + t));
        }
        
        acts.put("dWxr", dWxr);
        acts.put("dWdr", dWdr);
        acts.put("dWhr", dWhr);
        acts.put("dbr", dbr);
        
        acts.put("dWxz", dWxz);
        acts.put("dWdz", dWdz);
        acts.put("dWhz", dWhz);
        acts.put("dbz", dbz);
        
        acts.put("dWxh", dWxh);
        acts.put("dWdh", dWdh);
        acts.put("dWhh", dWhh);
        acts.put("dbh", dbh);
        
        acts.put("dWhd", dWhd);
        acts.put("dbd", dbd);
        
        acts.put("dWhy", dWhy);
        acts.put("dby", dby);
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	GRUBatchDerivative batchDerv = (GRUBatchDerivative) derv;
    	
        hdWxr = hdWxr.add(MatrixFunctions.pow(batchDerv.dWxr, 2.));
        hdWdr = hdWdr.add(MatrixFunctions.pow(batchDerv.dWdr, 2.));
        hdWhr = hdWhr.add(MatrixFunctions.pow(batchDerv.dWhr, 2.));
        hdbr = hdbr.add(MatrixFunctions.pow(batchDerv.dbr, 2.));
        
        hdWxz = hdWxz.add(MatrixFunctions.pow(batchDerv.dWxz, 2.));
        hdWdz = hdWdz.add(MatrixFunctions.pow(batchDerv.dWdz, 2.));
        hdWhz = hdWhz.add(MatrixFunctions.pow(batchDerv.dWhz, 2.));
        hdbz = hdbz.add(MatrixFunctions.pow(batchDerv.dbz, 2.));
        
        hdWxh = hdWxh.add(MatrixFunctions.pow(batchDerv.dWxh, 2.));
        hdWdh = hdWdh.add(MatrixFunctions.pow(batchDerv.dWdh, 2.));
        hdWhh = hdWhh.add(MatrixFunctions.pow(batchDerv.dWhh, 2.));
        hdbh = hdbh.add(MatrixFunctions.pow(batchDerv.dbh, 2.));
        
        hdWhd = hdWhd.add(MatrixFunctions.pow(batchDerv.dWhd, 2.));
        hdbd = hdbd.add(MatrixFunctions.pow(batchDerv.dbd, 2.));
        
        hdWhy = hdWhy.add(MatrixFunctions.pow(batchDerv.dWhy, 2.));
        hdby = hdby.add(MatrixFunctions.pow(batchDerv.dby, 2.));
        
        Wxr = Wxr.sub(batchDerv.dWxr.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxr).add(eps),-1.).mul(lr)));
        Wdr = Wdr.sub(batchDerv.dWdr.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdr).add(eps),-1.).mul(lr)));
        Whr = Whr.sub(batchDerv.dWhr.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhr).add(eps),-1.).mul(lr)));
        br = br.sub(batchDerv.dbr.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbr).add(eps),-1.).mul(lr)));
        
        Wxz = Wxz.sub(batchDerv.dWxz.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxz).add(eps),-1.).mul(lr)));
        Wdz = Wdz.sub(batchDerv.dWdz.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdz).add(eps),-1.).mul(lr)));
        Whz = Whz.sub(batchDerv.dWhz.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhz).add(eps),-1.).mul(lr)));
        bz = bz.sub(batchDerv.dbz.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbz).add(eps),-1.).mul(lr)));
        
        Wxh = Wxh.sub(batchDerv.dWxh.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWxh).add(eps),-1.).mul(lr)));
        Wdh = Wdh.sub(batchDerv.dWdh.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWdh).add(eps),-1.).mul(lr)));
        Whh = Whh.sub(batchDerv.dWhh.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhh).add(eps),-1.).mul(lr)));
        bh = bh.sub(batchDerv.dbh.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbh).add(eps),-1.).mul(lr)));

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
    	
    	GRUBatchDerivative batchDerv = (GRUBatchDerivative) derv;

		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		hd2Wxr = hd2Wxr.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxr, 2.).mul(1 - beta2));
		hd2Wdr = hd2Wdr.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdr, 2.).mul(1 - beta2));
		hd2Whr = hd2Whr.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhr, 2.).mul(1 - beta2));
		hd2br = hd2br.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbr, 2.).mul(1 - beta2));
		
		hdWxr = hdWxr.mul(beta1).add(batchDerv.dWxr.mul(1 - beta1));
		hdWdr = hdWdr.mul(beta1).add(batchDerv.dWdr.mul(1 - beta1));
		hdWhr = hdWhr.mul(beta1).add(batchDerv.dWhr.mul(1 - beta1));
		hdbr = hdbr.mul(beta1).add(batchDerv.dbr.mul(1 - beta1));

		hd2Wxz = hd2Wxz.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxz, 2.).mul(1 - beta2));
		hd2Wdz = hd2Wdz.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdz, 2.).mul(1 - beta2));
		hd2Whz = hd2Whz.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhz, 2.).mul(1 - beta2));
		hd2bz = hd2bz.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbz, 2.).mul(1 - beta2));
		
		hdWxz = hdWxz.mul(beta1).add(batchDerv.dWxz.mul(1 - beta1));
		hdWdz = hdWdz.mul(beta1).add(batchDerv.dWdz.mul(1 - beta1));
		hdWhz = hdWhz.mul(beta1).add(batchDerv.dWhz.mul(1 - beta1));
		hdbz = hdbz.mul(beta1).add(batchDerv.dbz.mul(1 - beta1));

		hd2Wxh = hd2Wxh.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWxh, 2.).mul(1 - beta2));
		hd2Wdh = hd2Wdh.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWdh, 2.).mul(1 - beta2));
		hd2Whh = hd2Whh.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhh, 2.).mul(1 - beta2));
		hd2bh = hd2bh.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbh, 2.).mul(1 - beta2));
		
		hdWxh = hdWxh.mul(beta1).add(batchDerv.dWxh.mul(1 - beta1));
		hdWdh = hdWdh.mul(beta1).add(batchDerv.dWdh.mul(1 - beta1));
		hdWhh = hdWhh.mul(beta1).add(batchDerv.dWhh.mul(1 - beta1));
		hdbh = hdbh.mul(beta1).add(batchDerv.dbh.mul(1 - beta1));

		hd2Whd = hd2Whd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhd, 2.).mul(1 - beta2));
		hd2bd = hd2bd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbd, 2.).mul(1 - beta2));
		
		hdWhd = hdWhd.mul(beta1).add(batchDerv.dWhd.mul(1 - beta1));
		hdbd = hdbd.mul(beta1).add(batchDerv.dbd.mul(1 - beta1));
		
		hd2Why = hd2Why.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhy, 2.).mul(1 - beta2));
		hd2by = hd2by.mul(beta2).add(MatrixFunctions.pow(batchDerv.dby, 2.).mul(1 - beta2));
		
		hdWhy = hdWhy.mul(beta1).add(batchDerv.dWhy.mul(1 - beta1));
		hdby = hdby.mul(beta1).add(batchDerv.dby.mul(1 - beta1));

		Wxr = Wxr.sub(
					hdWxr.mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxr.mul(biasBeta2)).add(eps), -1))
					);
		Wdr = Wdr.sub(
				hdWdr.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdr.mul(biasBeta2)).add(eps), -1))
				);
		Whr = Whr.sub(
				hdWhr.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whr.mul(biasBeta2)).add(eps), -1))
				);
		br = br.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2br.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbr.mul(biasBeta1)).mul(lr)
				);
		
		Wxz = Wxz.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxz.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWxz.mul(biasBeta1)).mul(lr)
				);
		Wdz = Wdz.sub(
				hdWdz.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdz.mul(biasBeta2)).add(eps), -1))
				);
		Whz = Whz.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whz.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWhz.mul(biasBeta1)).mul(lr)
				);
		bz = bz.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bz.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbz.mul(biasBeta1)).mul(lr)
				);

		Wxh = Wxh.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wxh.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWxh.mul(biasBeta1)).mul(lr)
				);
		Wdh = Wdh.sub(
				hdWdh.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Wdh.mul(biasBeta2)).add(eps), -1))
				);
		Whh = Whh.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whh.mul(biasBeta2)).add(eps), -1.)
				.mul(hdWhh.mul(biasBeta1)).mul(lr)
				);
		bh = bh.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bh.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbh.mul(biasBeta1)).mul(lr)
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
    
    public DoubleMatrix dDecode (DoubleMatrix ht) {
        return ht.mmul(Whd).add(bd);
    }

	/* (non-Javadoc)
	 * @see com.kingwang.ctsrnn.rnn.RNNCell#yDecode(org.jblas.DoubleMatrix)
	 */
	@Override
	public DoubleMatrix yDecode(DoubleMatrix ht) {
		return ht.mmul(Why).add(by);
	}

	/* (non-Javadoc)
	 * @see com.kingwang.ctsrnn.lstm.RNNCell#loadRNNModel(java.lang.String)
	 */
	@Override
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
	    			case Wxr: this.Wxr = matrixSetter(row, elems, this.Wxr); break;
	    			case Wdr: this.Wdr = matrixSetter(row, elems, this.Wdr); break;
	    			case Whr: this.Whr = matrixSetter(row, elems, this.Whr); break;
	    			case br: this.br = matrixSetter(row, elems, this.br); break;
	    			
	    			case Wxz: this.Wxz = matrixSetter(row, elems, this.Wxz); break;
	    			case Wdz: this.Wdz = matrixSetter(row, elems, this.Wdz); break;
	    			case Whz: this.Whz = matrixSetter(row, elems, this.Whz); break;
	    			case bz: this.bz = matrixSetter(row, elems, this.bz); break;
	    			
	    			case Wxh: this.Wxh = matrixSetter(row, elems, this.Wxh); break;
	    			case Wdh: this.Wdh = matrixSetter(row, elems, this.Wdh); break;
	    			case Whh: this.Whh = matrixSetter(row, elems, this.Whh); break;
	    			case bh: this.bh = matrixSetter(row, elems, this.bh); break;
	    			
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

	/* (non-Javadoc)
	 * @see com.kingwang.ctsrnn.lstm.RNNCell#writeRes(java.lang.String)
	 */
	@Override
	public void writeRes(String outFile) {
		OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile);
    	FileUtil.writeln(osw, "Wxr");
    	writeMatrix(osw, Wxr);
    	FileUtil.writeln(osw, "Wdr");
    	writeMatrix(osw, Wdr);
    	FileUtil.writeln(osw, "Whr");
    	writeMatrix(osw, Whr);
    	FileUtil.writeln(osw, "br");
    	writeMatrix(osw, br);
    	
    	FileUtil.writeln(osw, "Wxz");
    	writeMatrix(osw, Wxz);
    	FileUtil.writeln(osw, "Wdz");
    	writeMatrix(osw, Wdz);
    	FileUtil.writeln(osw, "Whz");
    	writeMatrix(osw, Whz);
    	FileUtil.writeln(osw, "bz");
    	writeMatrix(osw, bz);
    	
    	FileUtil.writeln(osw, "Wxh");
    	writeMatrix(osw, Wxh);
    	FileUtil.writeln(osw, "Wdh");
    	writeMatrix(osw, Wdh);
    	FileUtil.writeln(osw, "Whh");
    	writeMatrix(osw, Whh);
    	FileUtil.writeln(osw, "bh");
    	writeMatrix(osw, bh);
    	
    	FileUtil.writeln(osw, "Whd");
    	writeMatrix(osw, Whd);
    	FileUtil.writeln(osw, "bd");
    	writeMatrix(osw, bd);
    	
    	FileUtil.writeln(osw, "Why");
    	writeMatrix(osw, Why);
    	FileUtil.writeln(osw, "by");
    	writeMatrix(osw, by);
	}
    
}
