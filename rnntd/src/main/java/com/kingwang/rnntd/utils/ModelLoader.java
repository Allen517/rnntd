/**   
 * @package	com.kingwang.rnncdm.utils
 * @File		ModelLoader.java
 * @Crtdate	May 22, 2016
 *
 * Copyright (c) 2016 by <a href="mailto:wangyongqing.casia@gmail.com">King Wang</a>.   
 */
package com.kingwang.rnntd.utils;

import java.io.BufferedReader;
import java.io.IOException;

import org.jblas.DoubleMatrix;

import com.kingwang.rnntd.comm.utils.FileUtil;

/**
 *
 * @author King Wang
 * 
 * May 22, 2016 5:03:55 PM
 * @version 1.0
 */
public class ModelLoader {
	
	private DoubleMatrix Wxi;
	private DoubleMatrix Wdi;
    private DoubleMatrix Whi;
    private DoubleMatrix Wci;
    private DoubleMatrix bi;
    
    private DoubleMatrix Wxf;
    private DoubleMatrix Wdf;
    private DoubleMatrix Whf;
    private DoubleMatrix Wcf;
    private DoubleMatrix bf;
    
    private DoubleMatrix Wxc;
    private DoubleMatrix Wdc;
    private DoubleMatrix Whc;
    private DoubleMatrix bc;
    
    private DoubleMatrix Wxo;
    private DoubleMatrix Wdo;
    private DoubleMatrix Who;
    private DoubleMatrix Wco;
    private DoubleMatrix bo;
    
    private DoubleMatrix Whd;
    private DoubleMatrix bd;
    
    public ModelLoader(int inSize, int inTmSize, int outSize, int nodeSize) {
    	init(inSize, inTmSize, outSize, nodeSize);
    }
    
    private void init(int inSize, int inTmSize, int outSize, int nodeSize) {
    	this.Wxi = new DoubleMatrix(inSize, outSize);
    	this.Wdi = new DoubleMatrix(inTmSize, outSize);
        this.Whi = new DoubleMatrix(outSize, outSize);
        this.Wci = new DoubleMatrix(outSize, outSize);
        this.bi = new DoubleMatrix(1, outSize);
        
        this.Wxf = new DoubleMatrix(inSize, outSize);
        this.Wdf = new DoubleMatrix(inTmSize, outSize);
        this.Whf = new DoubleMatrix(outSize, outSize);
        this.Wcf = new DoubleMatrix(outSize, outSize);
        this.bf = new DoubleMatrix(1, outSize);
        
        this.Wxc = new DoubleMatrix(inSize, outSize);
        this.Wdc = new DoubleMatrix(inTmSize, outSize);
        this.Whc = new DoubleMatrix(outSize, outSize);
        this.bc = new DoubleMatrix(1, outSize);
        
        this.Wxo = new DoubleMatrix(inSize, outSize);
        this.Wdo = new DoubleMatrix(inTmSize, outSize);
        this.Who = new DoubleMatrix(outSize, outSize);
        this.Wco = new DoubleMatrix(outSize, outSize);
        this.bo = new DoubleMatrix(1, outSize);
        
        this.Whd = new DoubleMatrix(outSize, nodeSize);
        this.bd = new DoubleMatrix(1, nodeSize);
    }
    
    private DoubleMatrix matrixSetter(int row, String[] elems, DoubleMatrix x) {
    	
    	int col = x.getColumns();
    	if(elems.length!=col) {
    		System.err.println("Matrix setter in ModelLoader meets problem: the column number in file" +
    				" is not equal to the number in matrix");
    		return DoubleMatrix.EMPTY;
    	}
    	
    	for(int k=0; k<elems.length; k++) {
    		x.put(row, k, Double.parseDouble(elems[k]));
    	}
    	
    	return x;
    }
    
    public void loadRNNModel(String rnnModelFile) {

    	LoadTypes type = LoadTypes.Null;
    	int row = 0;
    	
    	try(BufferedReader br = FileUtil.getBufferReader(rnnModelFile)) {
    		String line = null;
    		while((line=br.readLine())!=null) {
    			String[] elems = line.split(",");
    			if(elems.length<2) {
    				type = LoadTypes.valueOf(elems[0]);
    				row = 0;
    				continue;
    			}
    			switch(type) {
	    			case Wxi: Wxi = matrixSetter(row, elems, Wxi); break;
	    			case Wdi: Wdi = matrixSetter(row, elems, Wdi); break;
	    			case Whi: Whi = matrixSetter(row, elems, Whi); break;
	    			case Wci: Wci = matrixSetter(row, elems, Wci); break;
	    			case bi: bi = matrixSetter(row, elems, bi); break;
	    			
	    			case Wxf: Wxf = matrixSetter(row, elems, Wxf); break;
	    			case Wdf: Wdf = matrixSetter(row, elems, Wdf); break;
	    			case Whf: Whf = matrixSetter(row, elems, Whf); break;
	    			case Wcf: Wcf = matrixSetter(row, elems, Wcf); break;
	    			case bf: bf = matrixSetter(row, elems, bf); break;
	    			
	    			case Wxc: Wxc = matrixSetter(row, elems, Wxc); break;
	    			case Wdc: Wdc = matrixSetter(row, elems, Wdc); break;
	    			case Whc: Whc = matrixSetter(row, elems, Whc); break;
	    			case bc: bc = matrixSetter(row, elems, bc); break;
	    			
	    			case Wxo: Wxo = matrixSetter(row, elems, Wxo); break;
	    			case Wdo: Wdo = matrixSetter(row, elems, Wdo); break;
	    			case Who: Who = matrixSetter(row, elems, Who); break;
	    			case Wco: Wco = matrixSetter(row, elems, Wco); break;
	    			case bo: bo = matrixSetter(row, elems, bo); break;
	    			
	    			case Whd: Whd = matrixSetter(row, elems, Whd); break;
	    			case bd: bd= matrixSetter(row, elems, bd); break;
    			}
    			row++;
    		}
    		
    	} catch(IOException e) {
    		
    	}
    }
    
    /**
	 * @return the wxi
	 */
	public DoubleMatrix getWxi() {
		return Wxi;
	}

	/**
	 * @param wxi the wxi to set
	 */
	public void setWxi(DoubleMatrix wxi) {
		Wxi = wxi;
	}

	/**
	 * @return the whi
	 */
	public DoubleMatrix getWhi() {
		return Whi;
	}

	/**
	 * @param whi the whi to set
	 */
	public void setWhi(DoubleMatrix whi) {
		Whi = whi;
	}

	/**
	 * @return the wci
	 */
	public DoubleMatrix getWci() {
		return Wci;
	}

	/**
	 * @param wci the wci to set
	 */
	public void setWci(DoubleMatrix wci) {
		Wci = wci;
	}

	/**
	 * @return the bi
	 */
	public DoubleMatrix getBi() {
		return bi;
	}

	/**
	 * @param bi the bi to set
	 */
	public void setBi(DoubleMatrix bi) {
		this.bi = bi;
	}

	/**
	 * @return the wxf
	 */
	public DoubleMatrix getWxf() {
		return Wxf;
	}

	/**
	 * @param wxf the wxf to set
	 */
	public void setWxf(DoubleMatrix wxf) {
		Wxf = wxf;
	}

	/**
	 * @return the whf
	 */
	public DoubleMatrix getWhf() {
		return Whf;
	}

	/**
	 * @param whf the whf to set
	 */
	public void setWhf(DoubleMatrix whf) {
		Whf = whf;
	}

	/**
	 * @return the wcf
	 */
	public DoubleMatrix getWcf() {
		return Wcf;
	}

	/**
	 * @param wcf the wcf to set
	 */
	public void setWcf(DoubleMatrix wcf) {
		Wcf = wcf;
	}

	/**
	 * @return the bf
	 */
	public DoubleMatrix getBf() {
		return bf;
	}

	/**
	 * @param bf the bf to set
	 */
	public void setBf(DoubleMatrix bf) {
		this.bf = bf;
	}

	/**
	 * @return the wxc
	 */
	public DoubleMatrix getWxc() {
		return Wxc;
	}

	/**
	 * @param wxc the wxc to set
	 */
	public void setWxc(DoubleMatrix wxc) {
		Wxc = wxc;
	}

	/**
	 * @return the whc
	 */
	public DoubleMatrix getWhc() {
		return Whc;
	}

	/**
	 * @param whc the whc to set
	 */
	public void setWhc(DoubleMatrix whc) {
		Whc = whc;
	}

	/**
	 * @return the bc
	 */
	public DoubleMatrix getBc() {
		return bc;
	}

	/**
	 * @param bc the bc to set
	 */
	public void setBc(DoubleMatrix bc) {
		this.bc = bc;
	}

	/**
	 * @return the wxo
	 */
	public DoubleMatrix getWxo() {
		return Wxo;
	}

	/**
	 * @param wxo the wxo to set
	 */
	public void setWxo(DoubleMatrix wxo) {
		Wxo = wxo;
	}

	/**
	 * @return the who
	 */
	public DoubleMatrix getWho() {
		return Who;
	}

	/**
	 * @param who the who to set
	 */
	public void setWho(DoubleMatrix who) {
		Who = who;
	}

	/**
	 * @return the wco
	 */
	public DoubleMatrix getWco() {
		return Wco;
	}

	/**
	 * @param wco the wco to set
	 */
	public void setWco(DoubleMatrix wco) {
		Wco = wco;
	}

	/**
	 * @return the bo
	 */
	public DoubleMatrix getBo() {
		return bo;
	}

	/**
	 * @param bo the bo to set
	 */
	public void setBo(DoubleMatrix bo) {
		this.bo = bo;
	}

	/**
	 * @return the wdi
	 */
	public DoubleMatrix getWdi() {
		return Wdi;
	}

	/**
	 * @param wdi the wdi to set
	 */
	public void setWdi(DoubleMatrix wdi) {
		Wdi = wdi;
	}

	/**
	 * @return the wdf
	 */
	public DoubleMatrix getWdf() {
		return Wdf;
	}

	/**
	 * @param wdf the wdf to set
	 */
	public void setWdf(DoubleMatrix wdf) {
		Wdf = wdf;
	}

	/**
	 * @return the wdc
	 */
	public DoubleMatrix getWdc() {
		return Wdc;
	}

	/**
	 * @param wdc the wdc to set
	 */
	public void setWdc(DoubleMatrix wdc) {
		Wdc = wdc;
	}

	/**
	 * @return the wdo
	 */
	public DoubleMatrix getWdo() {
		return Wdo;
	}

	/**
	 * @param wdo the wdo to set
	 */
	public void setWdo(DoubleMatrix wdo) {
		Wdo = wdo;
	}

	/**
	 * @return the whd
	 */
	public DoubleMatrix getWhd() {
		return Whd;
	}

	/**
	 * @param whd the whd to set
	 */
	public void setWhd(DoubleMatrix whd) {
		Whd = whd;
	}

	/**
	 * @return the bd
	 */
	public DoubleMatrix getBd() {
		return bd;
	}

	/**
	 * @param by the by to set
	 */
	public void setBd(DoubleMatrix bd) {
		this.bd = bd;
	}

	public static void main(String[] args) {
    	
    	ModelLoader ml = new ModelLoader(100, 2, 50, 33);
    	ml.loadRNNModel("toy.nd32.res");
    	DoubleMatrix x = ml.getWhi();
    	for(int r=0; r<x.getRows(); r++) {
    		for(int c=0; c<x.getColumns(); c++) {
    			System.out.print(x.get(r, c)+",");
    		}
    		System.out.println("");
    	}
    }
}
