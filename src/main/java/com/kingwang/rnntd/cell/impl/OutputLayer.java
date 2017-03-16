/**   
 * @package	com.kingwang.cdmrnn.rnn
 * @File		OutputLayer.java
 * @Crtdate	Sep 28, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.cell.impl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.rnntd.batchderv.BatchDerivative;
import com.kingwang.rnntd.batchderv.impl.OutputBatchDerivative;
import com.kingwang.rnntd.cell.Cell;
import com.kingwang.rnntd.cell.Operator;
import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.utils.Activer;
import com.kingwang.rnntd.utils.LoadTypes;
import com.kingwang.rnntd.utils.MatIniter;
import com.kingwang.rnntd.utils.MatIniter.Type;

/**
 *
 * @author King Wang
 * 
 * Sep 28, 2016 5:00:51 PM
 * @version 1.0
 */
public class OutputLayer extends Operator implements Cell, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8868938450690252135L;
	
	public DoubleMatrix Why;
    public DoubleMatrix by;
    
    public DoubleMatrix Whd;
    public DoubleMatrix bd;
	
	public DoubleMatrix hdWhy;
    public DoubleMatrix hdby;
    
    public DoubleMatrix hdWhd;
    public DoubleMatrix hdbd;
	
	public DoubleMatrix hd2Why;
    public DoubleMatrix hd2by;
    
    public DoubleMatrix hd2Whd;
    public DoubleMatrix hd2bd;
    
    public DoubleMatrix w;
    
    public DoubleMatrix hdw;
    public DoubleMatrix hd2w;
    
    public OutputLayer(int outSize, int nodeSize, MatIniter initer) {
        if (initer.getType() == Type.Uniform) {
            this.Why = initer.uniform(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
            
            this.Whd = initer.uniform(outSize, nodeSize);
            this.bd = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.Gaussian) {
            this.Why = initer.gaussian(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
            
            this.Whd = initer.gaussian(outSize, nodeSize);
            this.bd = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if (initer.getType() == Type.SVD) {
            this.Why = initer.svd(outSize, nodeSize);
            this.by = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
            
            this.Whd = initer.svd(outSize, nodeSize);
            this.bd = new DoubleMatrix(1, nodeSize).add(AlgCons.biasInitVal);
        } else if(initer.getType() == Type.Test) {
        }
        
        this.hdWhy = new DoubleMatrix(outSize, nodeSize);
        this.hdby = new DoubleMatrix(1, nodeSize);
        
        this.hd2Why = new DoubleMatrix(outSize, nodeSize);
        this.hd2by = new DoubleMatrix(1, nodeSize);
        
        this.hdWhd = new DoubleMatrix(outSize, nodeSize);
        this.hdbd = new DoubleMatrix(1, nodeSize);
        
        this.hd2Whd = new DoubleMatrix(outSize, nodeSize);
        this.hd2bd = new DoubleMatrix(1, nodeSize);
        
        if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
        	w = initer.uniform(1, 1);
        	this.hdw = new DoubleMatrix(w.rows, w.columns);
        	this.hd2w = new DoubleMatrix(w.rows, w.columns);
        }
        
    }
    
    public void active(int t, Map<String, DoubleMatrix> acts, double... params) {
    	DoubleMatrix haty = yDecode(acts.get("h" + t));
        DoubleMatrix py = Activer.softmax(haty);
        acts.put("py" + t, py);
        
        DoubleMatrix d = dDecode(acts.get("h" + t));
        DoubleMatrix lambda = MatrixFunctions.exp(d);
        acts.put("lambda" + t, lambda);
    }
    
    public void bptt(Map<String, DoubleMatrix> acts, int lastT, Cell... cell) {
    	DoubleMatrix dWhy = new DoubleMatrix(Why.rows, Why.columns);
    	DoubleMatrix dby = new DoubleMatrix(by.rows, by.columns);
    	
    	DoubleMatrix dWhd = new DoubleMatrix(Whd.rows, Whd.columns);
    	DoubleMatrix dbd = new DoubleMatrix(bd.rows, bd.columns);
    	
    	DoubleMatrix dw = null;
    	if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
        	dw = new DoubleMatrix(w.rows, w.columns);
    	}
    	
    	DoubleMatrix tmList = acts.get("tmList");
    	DoubleMatrix ndList = acts.get("ndList");
    	for (int t = lastT; t > -1; t--) {
    		double tm = tmList.get(t);
    		int ndIdx = (int) ndList.get(t);
            // delta y
            DoubleMatrix py = acts.get("py" + t);
            DoubleMatrix y = acts.get("y" + t);
            
	        DoubleMatrix lambda = acts.get("lambda"+t);
        	DoubleMatrix deltaD = new DoubleMatrix(lambda.rows, lambda.columns); 
        	if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
        		deltaD = lambda.div(w).mul(MatrixFunctions.exp(w.mul(tm)).sub(1));
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
            			deltaD.put(k, lambda.get(k)*(tm-AlgCons.gamma));
            		} else {
            			deltaD.put(k, lambda.get(k)*(tm+AlgCons.gamma));
            		}
            	}
        	}
            
        	DoubleMatrix deltaY = py.sub(y);
            acts.put("dy" + t, deltaY);

            dWhy = dWhy.add(acts.get("h" + t).transpose().mmul(deltaY));
            dby = dby.add(deltaY);
            
            deltaD.put(ndIdx, deltaD.get(ndIdx)-1);
            acts.put("dd" + t, deltaD);
            
            dWhd = dWhd.add(acts.get("h" + t).transpose().mmul(acts.get("dd" + t)));
            dbd = dbd.add(acts.get("dd" + t));
            
            //delta w
            if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
            	dw = dw.add(MatrixFunctions.pow(w, -1).mul(tm).add(
            			MatrixFunctions.exp(w.mul(tm)).sub(1)
            			.mul(MatrixFunctions.pow(w, -1).mul(tm).sub(MatrixFunctions.pow(w, -2)))
            			).mul(lambda.sum())
            			.sub(tm)
            			);
            }
    	}
    	
    	acts.put("dWhy", dWhy);
    	acts.put("dby", dby);
    	
    	acts.put("dWhd", dWhd);
    	acts.put("dbd", dbd);
    	
    	if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
    		acts.put("dw", dw);
    	}
    }
    
    public void updateParametersByAdaGrad(BatchDerivative derv, double lr) {
    	
    	OutputBatchDerivative batchDerv = (OutputBatchDerivative) derv;
    	
        hdWhy = hdWhy.add(MatrixFunctions.pow(batchDerv.dWhy, 2.));
        hdby = hdby.add(MatrixFunctions.pow(batchDerv.dby, 2.));
        
        hdWhd = hdWhd.add(MatrixFunctions.pow(batchDerv.dWhd, 2.));
        hdbd = hdbd.add(MatrixFunctions.pow(batchDerv.dbd, 2.));
        
        
        Why = Why.sub(batchDerv.dWhy.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhy).add(eps),-1.).mul(lr)));
        by = by.sub(batchDerv.dby.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdby).add(eps),-1.).mul(lr)));
        
        Whd = Whd.sub(batchDerv.dWhd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdWhd).add(eps),-1.).mul(lr)));
        bd = bd.sub(batchDerv.dbd.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdbd).add(eps),-1.).mul(lr)));

        if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
        	hdw = hdw.add(MatrixFunctions.pow(batchDerv.dw, 2.));
        	w = w.sub(batchDerv.dw.mul(
        			MatrixFunctions.pow(MatrixFunctions.sqrt(hdw).add(eps), -1).mul(lr)));
        }
    }
    
    public void updateParametersByAdam(BatchDerivative derv, double lr
    						, double beta1, double beta2, int epochT) {
    	
    	OutputBatchDerivative batchDerv = (OutputBatchDerivative) derv;
    	
		double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		hd2Why = hd2Why.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhy, 2.).mul(1 - beta2));
		hd2by = hd2by.mul(beta2).add(MatrixFunctions.pow(batchDerv.dby, 2.).mul(1 - beta2));
		
		hdWhy = hdWhy.mul(beta1).add(batchDerv.dWhy.mul(1 - beta1));
		hdby = hdby.mul(beta1).add(batchDerv.dby.mul(1 - beta1));
		
		hd2Whd = hd2Whd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dWhd, 2.).mul(1 - beta2));
		hd2bd = hd2bd.mul(beta2).add(MatrixFunctions.pow(batchDerv.dbd, 2.).mul(1 - beta2));
		
		hdWhd = hdWhd.mul(beta1).add(batchDerv.dWhd.mul(1 - beta1));
		hdbd = hdbd.mul(beta1).add(batchDerv.dbd.mul(1 - beta1));
		
		Why = Why.sub(
				hdWhy.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Why.mul(biasBeta2)).add(eps), -1))
				);
		by = by.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2by.mul(biasBeta2)).add(eps), -1.)
				.mul(hdby.mul(biasBeta1)).mul(lr)
				);
		
		Whd = Whd.sub(
				hdWhd.mul(biasBeta1).mul(lr)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2Whd.mul(biasBeta2)).add(eps), -1))
				);
		bd = bd.sub(
				MatrixFunctions.pow(MatrixFunctions.sqrt(hd2bd.mul(biasBeta2)).add(eps), -1.)
				.mul(hdbd.mul(biasBeta1)).mul(lr)
				);

		if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
			hdw = hdw.mul(beta1).add(batchDerv.dw.mul(1 - beta1));
			hd2w = hd2w.mul(beta2).add(MatrixFunctions.pow(batchDerv.dw, 2).mul(1 - beta2));
			
			w = w.sub(
					hdw.mul(biasBeta1).mul(lr)
					.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2w.mul(biasBeta2)).add(eps), -1))
					);
		}
    }
    
    public DoubleMatrix yDecode(DoubleMatrix ht) {
		return ht.mmul(Why).add(by);
	}
    
    public DoubleMatrix dDecode (DoubleMatrix ht) {
        return ht.mmul(Whd).add(bd);
    }

	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#writeCellParameter(java.lang.String, boolean)
	 */
	@Override
	public void writeCellParameter(String outFile, boolean isAttached) {
		OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile, isAttached);
    	FileUtil.writeln(osw, "Why");
    	writeMatrix(osw, Why);
    	FileUtil.writeln(osw, "by");
    	writeMatrix(osw, by);
    	FileUtil.writeln(osw, "Whd");
    	writeMatrix(osw, Whd);
    	FileUtil.writeln(osw, "bd");
    	writeMatrix(osw, bd);
    	if(AlgCons.tmDist.equalsIgnoreCase("exp")) {
    		FileUtil.writeln(osw, "w");
    		writeMatrix(osw, w);
    	}
	}

	/* (non-Javadoc)
	 * @see com.kingwang.cdmrnn.rnn.Cell#loadCellParameter(java.lang.String)
	 */
	@Override
	public void loadCellParameter(String cellParamFile) {
		LoadTypes type = LoadTypes.Null;
		int row = 0;
		
		try(BufferedReader br = FileUtil.getBufferReader(cellParamFile)) {
			String line = null;
			while((line=br.readLine())!=null) {
				String[] elems = line.split(",");
				if(elems.length<2 && !elems[0].contains(".")) {
    				String typeStr = "Null";
    				String[] typeList = {"Why", "by", "Whd", "bd", "w"};
    				for(String tStr : typeList) {
    					if(elems[0].equalsIgnoreCase(tStr)) {
    						typeStr = tStr;
    						break;
    					}
    				}
    				type = LoadTypes.valueOf(typeStr);
    				row = 0;
    				continue;
    			}
				switch(type) {
					case Why: this.Why = matrixSetter(row, elems, this.Why); break;
					case by: this.by = matrixSetter(row, elems, this.by); break;
					case Whd: this.Whd = matrixSetter(row, elems, this.Whd); break;
					case bd: this.bd = matrixSetter(row, elems, this.bd); break;
					case w: this.w = matrixSetter(row, elems, this.w); break;
				}
				row++;
			}
			
		} catch(IOException e) {
			
		}
	}
}
