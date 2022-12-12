import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Pairwise Comparison Matrix untuk criteria dengan criteria
crxcr = np.array([[[1.,1.,1.],    [3.,4.,5],    [2.,3.,4.],    [4.,5.,6.]],
                  [[.2,.25,.5],   [1.,1.,1.],   [.25,.33,.50], [1.,2.,3.]],
                  [[.25,.33,.25], [2.,3.,4.],   [1.,1.,1.],    [2.,3.,4.]],
                  [[.17,.2,.25],  [.33,.50,1.], [.25,.33,.5],  [1.,1.,1.]]])

criteriaDict = {
            1: "Curah Hujan",
            2: "Kelembaban Udara",
            3: "Suhu",
            4: "Ketinggian Tanah"}

alternativesName =[
    "Padi",        "Jagung",      "Kedelai",    "Kacang Tanah", "Kacang Hijau", "Ubi Kayu",     "Ubi Jalar" ]
    
#Pairwise Comparison Matrix untuk alternatif dengan alternatif berdasarkan criteria 1 (Curah Hujan)
altxalt_cr1 = np.array([
    [[1.,1.,1.],   [2.,3.,4.],   [1.,2.,3.],   [2.,3.,4.],   [3,4,5],     [1.,2.,3.],   [4.,5.,6.]  ],
    [[.25,.33,.50],[1.,1.,1.],   [.33,.5,1.],  [1.,1.,1.],   [1.,2.,3.],  [.33,.5,1.],  [2.,3.,4]   ],
    [[.33,.50,1.], [1.,2.,3.],   [1.,1.,1.],   [1.,2.,3.],   [2.,3.,4],   [1.,1.,1.],   [3.,4.,5]   ],
    [[.25,.33,.50],[1.,1.,1.],   [.33,.5,1.],  [1.,1.,1.],   [1.,2.,3.],  [.33,.5,1.],  [2.,3.,4]   ],
    [[.20,.25,.33],[.33,.50,1.], [.25,.33,.50],[.33,.50,1.], [1.,1.,1.],  [.25,.33,.50],[1.,2.,3.]  ],
    [[.33,.50,1.], [1.,2.,3.],   [1.,1.,1.],   [1.,2.,3.],   [2.,3.,4],   [1.,1.,1.],   [3.,4.,5]   ],
    [[.17,.20,.25],[.25,.33,.50],[.20,.25,.33],[.25,.33,.50],[.33,.50,1.],[.20,.25,.33],[1.,1.,1]   ]])
#Pairwise Comparison Matrix untuk alternatif dengan alternatif berdasarkan criteria 2 (Kelembaban Udara)
altxalt_cr2 = np.array([
    [[1.,1.,1.],   [1.,1.,1.],   [3.,4.,5.],   [2.,3.,4.],   [1.,2.,3.],  [2.,3.,4.],   [1.,2.,3.]  ],
    [[1.,1.,1.],   [1.,1.,1.],   [3.,4.,5.],   [2.,3.,4.],   [1.,2.,3.],  [2.,3.,4.],   [1.,2.,3.]  ],
    [[.20,.25,.33],[.20,.25,.33],[1.,1.,1.],   [.33,.5,1.],  [.33,.5,1.], [1.,1.,1.],   [.25,.33,.5]],
    [[.25,.33,.50],[.25,.33,.50],[1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],  [1.,2.,3.],   [1.,1.,1.]  ],
    [[.33,.50,1.], [.33,.50,1.], [1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],  [1.,2.,3.],   [1.,1.,1]   ],
    [[.25,.33,.50],[.25,.33,.50],[1.,1.,1.],   [.33,.5,1.],  [.33,.5,1.], [1.,1.,1.],   [.33,.5,1]  ],
    [[.33,.50,1.], [.33,.50,1.], [2.,3.,4.],   [1.,1.,1.],   [1.,1.,1.],  [1.,2.,3.],   [1.,1.,1]   ]])
#Pairwise Comparison Matrix untuk alternatif dengan alternatif berdasarkan criteria 3 (Suhu)
altxalt_cr3 = np.array([
    [[1.,1.,1.],   [.33,.50,1.], [.33,.50,1.], [1.,1.,1.],   [3.,4.,5.],  [1.,2.,3.],   [2.,3.,4.]  ],
    [[1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [1.,2.,3.],   [4.,5.,6],   [2.,3.,4.],   [3.,4.,5]   ],
    [[1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [1.,2.,3.],   [4.,5.,6],   [2.,3.,4.],   [3.,4.,5]   ],
    [[1.,1.,1.],   [.33,.50,1.], [.33,.50,1.], [1.,1.,1.],   [3.,4.,5.],  [1.,2.,3.],   [2.,3.,4.]  ],
    [[.20,.25,.33],[.17,.20,.25],[.17,.20,.25],[.20,.25,.33],[1.,1.,1.],  [.25,.33,.50],[.33,.5,1]  ],
    [[.33,.50,1.], [.25,.33,.50],[.25,.33,.50],[.33,.50,1.], [2.,3.,4.],  [1.,1.,1.],   [1.,2.,3.]  ],
    [[.25,.33,.50],[.20,.25,.33],[.20,.25,.33],[.25,.33,.50],[1.,2.,3.],  [.33,.50,1.], [1.,1.,1]   ]])
#Pairwise Comparison Matrix untuk alternatif dengan alternatif berdasarkan criteria 4 (Ketinggian Tanah) daerah kelompok 1
altxalt_cr4_k1 = np.array([
    [[1.,1.,1.],   [1.,1.,1.],   [1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [1.,1.,1.],   [4.,5.,6.]   ],
    [[1.,1.,1.],   [1.,1.,1.],   [1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [1.,1.,1.],   [4.,5.,6.]   ],
    [[.33,.50,1.], [.33,.50,1.], [1.,1.,1.],   [.33,.50,1.], [.33,.50,1.], [.33,.50,1.], [2.,3.,4.]   ],
    [[1.,1.,1.],   [1.,1.,1.],   [1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [1.,1.,1.],   [4.,5.,6.]   ],
    [[1.,1.,1.],   [1.,1.,1.],   [1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [1.,1.,1.],   [4.,5.,6.]   ],
    [[1.,1.,1.],   [1.,1.,1.],   [1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [1.,1.,1.],   [4.,5.,6.]   ],
    [[.17,.20,.25],[.17,.20,.25],[.25,.33,.50],[.17,.20,.25],[.17,.20,.25],[.17,.20,.25],[1.,1.,1]    ]])
#Pairwise Comparison Matrix untuk alternatif dengan alternatif berdasarkan criteria 4 (Ketinggian Tanah) daerah kelompok 2
altxalt_cr4_k2 = np.array([
    [[1.,1.,1.],   [1.,1.,1.],   [2.,3.,4.],   [1.,2.,3.],   [1.,2.,3.],   [1.,1.,1.],   [1.,2.,3.]   ],
    [[1.,1.,1.],   [1.,1.,1.],   [2.,3.,4.],   [1.,2.,3.],   [1.,2.,3.],   [1.,1.,1.],   [1.,2.,3.]   ],
    [[.25,.33,.50],[.25,.33,.50],[1.,1.,1.],   [.33,.50,1.], [.33,.50,1.], [.25,.33,.50],[.33,.50,1.] ],
    [[.33,.50,1.], [.33,.50,1.], [1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [.25,.33,.50],[.33,.50,1.] ],
    [[.33,.50,1.], [.33,.50,1.], [1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [.25,.33,.50],[.33,.50,1.] ],
    [[1.,1.,1.],   [1.,1.,1.],   [2.,3.,4.],   [2.,3.,4.],   [2.,3.,4.],   [1.,1.,1.],   [1.,2.,3.]   ],
    [[.33,.50,1.], [.33,.50,1.], [1.,2.,3.],   [1.,2.,3.],   [1.,2.,3.],   [.33,.50,1.], [1.,1.,1]    ]])
#Pairwise Comparison Matrix untuk alternatif dengan alternatif berdasarkan criteria 4 (Ketinggian Tanah) daerah kelompok 3
altxalt_cr4_k3 = np.array([
    [[1.,1.,1.],   [1.,1.,1.],   [4.,5.,6.],   [2.,3.,4.],   [2.,3.,4.],   [1.,2.,3.],   [1.,1.,1.]   ],
    [[1.,1.,1.],   [1.,1.,1.],   [4.,5.,6.],   [2.,3.,4.],   [2.,3.,4.],   [1.,2.,3.],   [1.,1.,1.]   ],
    [[.17,.20,.25],[.17,.20,.25],[1.,1.,1.],   [.25,.33,.50],[.25,.33,.50],[.20,.25,.33],[.17,.20,.25]],
    [[.25,.33,.50],[.25,.33,.50],[2.,3.,4.],   [1.,1.,1.],   [1.,1.,1.],   [.33,.50,1.], [.25,.33,.50]],
    [[.25,.33,.50],[.25,.33,.50],[2.,3.,4.],   [1.,1.,1.],   [1.,1.,1.],   [.33,.50,1.], [.25,.33,.50]],
    [[.33,.50,1.], [.33,.50,1.], [3.,4.,5.],   [1.,2.,3.],   [1.,2.,3.],   [1.,1.,1.],   [.33,.50,1.] ],
    [[1.,1.,1.],   [1.,1.,1.],   [4.,5.,6.],   [2.,3.,4.],   [2.,3.,4.],   [1.,2.,3.],   [1.,1.,1.]   ]])
#Pairwise Comparison Matrix untuk alternatif dengan alternatif berdasarkan criteria 4 (Ketinggian Tanah) daerah kelompok 4
altxalt_cr4_k4 = np.array([
    [[1,1.,1],     [1.,2.,3.],   [5.,6.,7.],   [4.,5.,6.],   [4.,5.,6.],   [3.,4.,5.],   [2.,3.,4]    ],
    [[.33,.50,1.], [1.,1,1.],    [4.,5.,6.],   [3.,4.,5.],   [3.,4.,5.],   [2.,3.,4.],   [1.,2.,3]    ],
    [[.14,.17,.20],[.17,.20,.25],[1.,1,1.],    [.33,.5,1.],  [.33,.5,1.],  [.25,.33,.5], [.2,.25,.33] ],
    [[.17,.20,.25],[.20,.25,.33],[1.,2.,3.],   [1.,1.,1.],   [1.,1.,1],    [.33,.5,1.],  [.25,.33,.5] ],
    [[.17,.20,.25],[.20,.25,.33],[1.,2.,3.],   [1.,1.,1.],   [1.,1,1.],    [.33,.5,1.],  [.25,.33,.5] ],
    [[.20,.25,.33],[.25,.33,.50],[2.,3.,4.],   [1.,2.,3.],   [1.,2.,3.],   [1.,1,1.],    [.33,.5,1.]  ],
    [[.25,.33,.50],[.33,.50,1.], [3.,4.,5.],   [2.,3.,4.],   [2.,3.,4.],   [1.,2.,3.],   [1.,1.,1.]   ]])
#Pairwise Comparison Matrix untuk alternatif dengan alternatif berdasarkan criteria 4 (Ketinggian Tanah) daerah kelompok 5
altxalt_cr4_k5 = np.array([
    [[1,1.,1],     [2.,3.,4.],   [6.,7.,8.],   [5.,6.,7.],   [5.,6.,7.],   [4.,5.,6.],   [3.,4.,5]    ],
    [[.25,.33,.50],[1.,1,1.],    [5.,6.,7.],   [4.,5.,6.],   [4.,5.,6.],   [3.,4.,5.],   [2.,3.,4]    ],
    [[.13,.14,.17],[.14,.17,.20],[1.,1,1.],    [.33,.5,1.],  [.33,.5,1.],  [.25,.33,.5], [.17,.2,.25] ],
    [[.14,.17,.20],[.17,.20,.25],[1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [.33,.5,1.],  [.25,.33,.5] ],
    [[.14,.17,.20],[.17,.20,.25],[1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [.33,.5,1.],  [.25,.33,.5] ],
    [[.17,.20,.25],[.20,.25,.33],[2.,3.,4.],   [1.,2.,3.],   [1.,2.,3.],   [1.,1,1.],    [.33,.5,1]   ],
    [[.20,.25,.33],[.25,.33,.50],[4.,5.,6.],   [2.,3.,4.],   [2.,3.,4.],   [1.,2.,3.],   [1.,1,1]     ]])
#Pairwise Comparison Matrix untuk alternatif dengan alternatif berdasarkan criteria 4 (Ketinggian Tanah) daerah kelompok 6
altxalt_cr4_k6 = np.array([
    [[1,1.,1],     [2.,3.,4.],   [6.,7.,8.],   [5.,6.,7.],   [5.,6.,7.],   [4.,5.,6.],   [3.,4.,5]    ],
    [[.25,.33,.50],[1.,1,1.],    [5.,6.,7.],   [4.,5.,6.],   [4.,5.,6.],   [3.,4.,5.],   [2.,3.,4]    ],
    [[.13,.14,.17],[.14,.17,.20],[1.,1,1.],    [.33,.5,1.],  [.33,.5,1.],  [.25,.33,.5], [.17,.2,.25] ],
    [[.14,.17,.20],[.17,.20,.25],[1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [.33,.5,1.],  [.25,.33,.5] ],
    [[.14,.17,.20],[.17,.20,.25],[1.,2.,3.],   [1.,1.,1.],   [1.,1.,1.],   [.33,.5,1.],  [.25,.33,.5] ],
    [[.17,.20,.25],[.20,.25,.33],[2.,3.,4.],   [1.,2.,3.],   [1.,2.,3.],   [1.,1,1.],    [.33,.5,1]   ],
    [[.20,.25,.33],[.25,.33,.50],[4.,5.,6.],   [2.,3.,4.],   [2.,3.,4.],   [1.,2.,3.],   [1.,1,1]     ]])


#Param: matrix = Matrix yang akan dihitung konsistensinya, printComp = opsi untuk menampilkan komputasi konsistensi matrix
def isConsistent(matrix, printComp=True):
    RI = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }

    mat_len = len(matrix)
    midMatrix = np.zeros((mat_len, mat_len))
    #weights_sum = np.sum(matrix)
    for i in range(mat_len):
        for j in range(mat_len):
            midMatrix[i][j] = matrix[i][j][1]
    if(printComp): print("mid-value matrix: \n", midMatrix, "\n")

    eigenvalue = np.real(np.linalg.eigvals(midMatrix))
    lambdaMax = max(eigenvalue)
    if(printComp): print("eigenvalue: ", eigenvalue)
    if(printComp): print("lambdaMax: ", lambdaMax)
    if(printComp): print("\n")

    RIValue = RI[mat_len]
    if(printComp): print("R.I. Value: ", RIValue)

    CIValue = (lambdaMax-mat_len)/(mat_len - 1)
    if(printComp): print("C.I. Value: ", CIValue)


    CRValue = CIValue/RIValue
    if(printComp): print("C.R. Value: ", CRValue)

    if(printComp): print("\n")
    if(CRValue<=0.1):
        if(printComp): print("Matrix reasonably consistent, we could continue")
        return True
    else:
        if(printComp): print("Consistency Ratio is greater than 10%, we need to revise the subjective judgment")
        return False
    
    #Param: matrix = Matrix yang akan dihitung konsistensinya, printComp = opsi untuk menampilkan komputasi konsistensi matrix
def pairwiseComp(matrix, printComp=True):
    matrix_len = len(matrix)

    #calculate fuzzy geometric mean value
    geoMean = np.zeros((len(matrix),3))

    for i in range(matrix_len):
        for j in range(3):
            temp = 1
            for tfn in matrix[i]:
                temp *= tfn[j]
            temp = pow(temp, 1/matrix_len)
            geoMean[i,j] = temp
    
    if(printComp): print("Fuzzy Geometric Mean Value: \n", geoMean, "\n")
    #calculate the sum of fuzzy geometric mean value
    geoMean_sum = np.zeros(3)
    for row in geoMean:
        geoMean_sum[0] += row[0]
        geoMean_sum[1] += row[1]
        geoMean_sum[2] += row[2]
    
    if(printComp): print("Fuzzy Geometric Mean Sum:", geoMean_sum, "\n")
    #calculate weights
    weights = np.zeros(matrix_len)

    for i in range(len(geoMean)):
        temp = 0
        for j in range(len(geoMean[0])):
            temp += geoMean[i,j]*(1/geoMean_sum[(3-1)-j])
        weights[i] = temp 
    
    if(printComp): print("Weights: \n", weights, "\n")
    #caculate normaized weights
    normWeights = np.zeros(matrix_len)
    weights_sum = np.sum(weights)
    for i in range(matrix_len): 
        normWeights[i] = weights[i]/weights_sum
    
    if(printComp): print("Normalized Weights: ", normWeights,"\n")
    return normWeights

#Param: crxcr = Pairwise comparison matrix criteria X criteria, altxalt = Pairwise comparison matrices alternatif X alternatif , 
#       alternativesName = Nama dari setiap alternatif, printComp = opsi untuk menampilkan komputasi konsistensi matrix
def FAHP(crxcr, altxalt, alternativesName, printComp=True):
    crxcr_cons = isConsistent(crxcr, False)
    if(crxcr_cons):
        if(printComp): print("criteria X criteria comparison matrix reasonably consistent, we could continue")
    else: 
        if(printComp): print("criteria X criteria comparison matrix consistency ratio is greater than 10%, we need to revise the subjective judgment")
        
    for i, altxalt_cr in enumerate(altxalt):
        isConsistent(altxalt_cr, False)
        if(crxcr_cons):
            if(printComp): print("alternatives X alternatives comparison matrix for criteria",i+1," is reasonably consistent, we could continue")
        else: 
            if(printComp): print("alternatives X alternatives comparison matrix for criteria",i+1,"'s consistency ratio is greater than 10%, we need to revise the subjective judgment")
    
    if(printComp): print("\n")
    
    if(printComp): print("criteria X criteria ======================================================\n")
    crxcr_weights = pairwiseComp(crxcr, printComp)
    if(printComp): print("criteria X criteria weights: ", crxcr_weights)
    
    
    if(printComp): print("\n")
    if(printComp): print("alternative x alternative ======================================================\n")
    
    altxalt_weights = np.zeros((len(altxalt),len(altxalt[0])))
    for i, altxalt_cr in enumerate(altxalt):
        if(printComp): print("alternative x alternative for criteria", criteriaDict[(i+1)],"---------------\n")
        altxalt_weights[i] =  pairwiseComp(altxalt_cr, printComp)
        
    if(printComp): print("alternative x alternative weights:")
    altxalt_weights = altxalt_weights.transpose(1, 0)
    if(printComp): print(altxalt_weights)
    
    sumProduct = np.zeros(len(altxalt[0]))
    for i  in range(len(altxalt[0])):
        sumProduct[i] = np.dot(crxcr_weights, altxalt_weights[i])
        
    if(printComp): print("\n")
    if(printComp): print("RANKING =====================================================================\n")
    
    output_df = pd.DataFrame(data=[alternativesName, sumProduct]).T
    output_df = output_df.rename(columns={0: "Alternatives", 1: "Sum_of_Product"})
    output_df = output_df.sort_values(by=['Sum_of_Product'],ascending = False)
    output_df.index = np.arange(1,len(output_df)+1)
    
    if(printComp): print(output_df)
    
    return output_df

altxalt_k1 = np.stack((altxalt_cr1, altxalt_cr2, altxalt_cr3, altxalt_cr4_k1))
output_k1 = FAHP(crxcr, altxalt_k1, alternativesName, False)
print("Ranking Alternatif untuk kelompok 1:\n",output_k1,"\n")

altxalt_k2 = np.stack((altxalt_cr1, altxalt_cr2, altxalt_cr3, altxalt_cr4_k2))
output_k2 = FAHP(crxcr, altxalt_k2, alternativesName, False)
print("Ranking Alternatif untuk kelompok 2:\n",output_k2,"\n")

altxalt_k3 = np.stack((altxalt_cr1, altxalt_cr2, altxalt_cr3, altxalt_cr4_k3))
output_k3 = FAHP(crxcr, altxalt_k3, alternativesName, False)
print("Ranking Alternatif untuk kelompok 3:\n",output_k3,"\n")

altxalt_k4 = np.stack((altxalt_cr1, altxalt_cr2, altxalt_cr3, altxalt_cr4_k4))
output_k4 = FAHP(crxcr, altxalt_k4, alternativesName, False)
print("Ranking Alternatif untuk kelompok 4:\n",output_k4,"\n")

altxalt_k5 = np.stack((altxalt_cr1, altxalt_cr2, altxalt_cr3, altxalt_cr4_k5))
output_k5 = FAHP(crxcr, altxalt_k5, alternativesName, False)
print("Ranking Alternatif untuk kelompok 5:\n",output_k5,"\n")

altxalt_k6 = np.stack((altxalt_cr1, altxalt_cr2, altxalt_cr3, altxalt_cr4_k6))
output_k6 = FAHP(crxcr, altxalt_k6, alternativesName, False)
print("Ranking Alternatif untuk kelompok 6:\n",output_k6,"\n")