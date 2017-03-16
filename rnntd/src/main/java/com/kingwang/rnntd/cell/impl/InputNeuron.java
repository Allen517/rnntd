/**   
 * @package	com.kingwang.rnncdm.lstm
 * @File		InputNeuron.java
 * @Crtdate	Jun 18, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.cell.impl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.kingwang.rnntd.cell.Operator;
import com.kingwang.rnntd.cell.RNNCell;
import com.kingwang.rnntd.comm.utils.CollectionHelper;
import com.kingwang.rnntd.comm.utils.FileUtil;
import com.kingwang.rnntd.cons.AlgCons;
import com.kingwang.rnntd.utils.LoadTypes;
import com.kingwang.rnntd.utils.MatIniter;
import com.kingwang.rnntd.utils.MatIniter.Type;

/**
 * Data structure for input representation 
 *
 * @author King Wang
 * 
 * Jun 18, 2016 8:04:15 PM
 * @version 1.0
 */
public class InputNeuron extends Operator {

    public DoubleMatrix repMatrix;
    public List<String> nodeDict;
    
    public DoubleMatrix w;
    
    public DoubleMatrix tmFeat;
    
    public DoubleMatrix hdxMat;
    public DoubleMatrix hd2xMat;
    
    public DoubleMatrix hdw;
    public DoubleMatrix hd2w;
    
    public InputNeuron(int nodeSize, int repDim) {
    	init(new MatIniter(Type.Uniform));
    	repMatrix = new DoubleMatrix(nodeSize, repDim);
    	nodeDict = new ArrayList<>();
    }
    
    public InputNeuron(DoubleMatrix repMatrix, List<String> nodeDict) {
    	init(new MatIniter(Type.Uniform));
    	this.repMatrix = repMatrix; 
    	this.nodeDict = nodeDict;
    	this.hdxMat = new DoubleMatrix(repMatrix.rows, repMatrix.columns);
        this.hd2xMat = new DoubleMatrix(repMatrix.rows, repMatrix.columns);
        this.hdw = new DoubleMatrix(w.rows, w.columns);
        this.hd2w = new DoubleMatrix(w.rows, w.columns);
    }
    
    private void init(MatIniter initer) {
    	w = initer.uniform(1, 1);
    }
    
    /**
     * 
     * 
     * @param cell
     * @param nodes must be the input sequence
     * @param acts
     * @param lastT
     */
    public void bptt(RNNCell cell, List<Integer> nodes, List<Double> tmList
    					, Map<String, DoubleMatrix> acts, int lastT) {
    	
    	DoubleMatrix dxMat = new DoubleMatrix(repMatrix.rows, repMatrix.columns);
    	DoubleMatrix dw = new DoubleMatrix(w.rows, w.columns);
    	for (int t = 0; t < lastT + 1; t++) {
        	int ndIdx = nodes.get(t);
            
            //update input vectors
            DoubleMatrix dx = dxMat.getRow(ndIdx);
            if(AlgCons.rnnType.equalsIgnoreCase("lstm")) {
            	LSTM lstmCell = (LSTM) cell;
            	dx = clip(dx.add(acts.get("di"+t).mmul(lstmCell.Wxi.transpose())).add(acts.get("df"+t).mmul(lstmCell.Wxf.transpose()))
            			.add(acts.get("dgc"+t).mmul(lstmCell.Wxc.transpose())).add(acts.get("do"+t).mmul(lstmCell.Wxo.transpose())));
            }
            if(AlgCons.rnnType.equalsIgnoreCase("gru")) {
            	GRU gruCell = (GRU) cell;
            	dx = clip(dx.add(acts.get("dr"+t).mmul(gruCell.Wxr.transpose())).add(acts.get("dz"+t).mmul(gruCell.Wxz.transpose()))
            			.add(acts.get("dgh"+t).mmul(gruCell.Wxh.transpose())));
            }
            dxMat.putRow(ndIdx, dx);
            
            //delta w
            double tmGap = tmList.get(t);
            DoubleMatrix lambda = acts.get("lambda" + t); 
            dw = dw.add(MatrixFunctions.pow(w, -1).mul(tmGap).add(
            				MatrixFunctions.exp(w.mul(tmGap)).sub(1)
            				.mul(MatrixFunctions.pow(w, -1).mul(tmGap).sub(MatrixFunctions.pow(w, -2)))
            				).mul(lambda.sum())
            			.sub(tmGap)
            		);
        }
    	
    	acts.put("dxMat", dxMat);
    	acts.put("dw", dw);
    }
    
    public void updateParametersByAdaGrad(DoubleMatrix dxMat, DoubleMatrix dw, double lr_input) {
    	
    	hdxMat = hdxMat.add(MatrixFunctions.pow(dxMat, 2.));
    	hdw = hdw.add(MatrixFunctions.pow(dw, 2.));
    	
    	repMatrix = repMatrix.sub(dxMat.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdxMat).add(eps), -1).mul(lr_input)));
    	w = w.sub(dw.mul(
        		MatrixFunctions.pow(MatrixFunctions.sqrt(hdw).add(eps), -1).mul(lr_input)));
    }
    
    public void updateParametersByAdam(DoubleMatrix dxMat, DoubleMatrix dw, double lr_input
							, double beta1, double beta2, int epochT) {
    
    	double biasBeta1 = 1. / (1 - Math.pow(beta1, epochT));
		double biasBeta2 = 1. / (1 - Math.pow(beta2, epochT));

		hdxMat = hdxMat.mul(beta1).add(dxMat.mul(1 - beta1));
		hd2xMat = hd2xMat.mul(beta2).add(MatrixFunctions.pow(dxMat, 2).mul(1 - beta2));
		
		hdw = hdw.mul(beta1).add(dw.mul(1 - beta1));
		hd2w = hd2w.mul(beta2).add(MatrixFunctions.pow(dw, 2).mul(1 - beta2));
		
		repMatrix = repMatrix.sub(
				hdxMat.mul(biasBeta1).mul(lr_input)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2xMat.mul(biasBeta2)).add(eps), -1))
				);
		
		w = w.sub(
				hdw.mul(biasBeta1).mul(lr_input)
				.mul(MatrixFunctions.pow(MatrixFunctions.sqrt(hd2w.mul(biasBeta2)).add(eps), -1))
				);
    }
    
    public void writeRes(String outFile) {
    	
    	OutputStreamWriter osw = FileUtil.getOutputStreamWriter(outFile);
    	FileUtil.writeln(osw, "nodeDict");
    	for(int k=0; k<nodeDict.size(); k++) {
    		FileUtil.writeln(osw, nodeDict.get(k)+","+k);
    	}
    	FileUtil.writeln(osw, "repMat");
    	writeMatrix(osw, repMatrix);
    	FileUtil.writeln(osw, "w");
    	writeMatrix(osw, w);
    }
    
    private List<String> nodeDictSetter(String[	] elems, List<String> nodeDict) {
    	
    	if(CollectionHelper.isEmpty(nodeDict)) {
    		nodeDict = new ArrayList<>();
    	}
    	int idx = Integer.parseInt(elems[1]);
    	if(idx!=nodeDict.size()) {
    		System.out.println("Load nodeDict error!");
    		return Collections.emptyList();
    	}
    	nodeDict.add(elems[0]);
    	
    	return nodeDict;
    }
    
    public void loadRepresentation(String repFile) {

    	LoadTypes type = LoadTypes.Null;
    	int row = 0;
    	
    	try(BufferedReader br = FileUtil.getBufferReader(repFile)) {
    		String line = null;
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(",");
    			if(line.contains("b") || line.contains("W") || line.contains("repMat")
    					|| line.contains("nodeDict") || line.contains("w")) {
    				type = LoadTypes.valueOf(elems[0]);
    				row = 0;
    				continue;
    			}
    			switch(type) {
	    			case repMat: this.repMatrix = matrixSetter(row, elems, this.repMatrix); break;
	    			case nodeDict: this.nodeDict = nodeDictSetter(elems, this.nodeDict); break;
	    			case w: this.w = matrixSetter(row, elems, this.w); break;
    			}
    			row++;
    		}
    		
    	} catch(IOException e) {
    		
    	}
    }
}
