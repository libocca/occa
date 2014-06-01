#ifndef OCL_PREPROCESSOR_HEADER
#define OCL_PREPROCESSOR_HEADER

//---[ ESSENTIALS ]-------------------------------------------------------------
#define OCL_STRINGIFY2(...) #__VA_ARGS__
#define OCL_STRINGIFY(STR)  OCL_STRINGIFY2(STR)

#define OCL_PRINT_MACRO(X) std::cout << OCL_STRINGIFY(X) << endl;

#define OCL_PCAT(LEFT, ...) LEFT ## __VA_ARGS__
#define OCL_CAT(LEFT, ...)  OCL_PCAT(LEFT, __VA_ARGS__)

#define OCL_EXPR(...) __VA_ARGS__

#define OCL_VOID_MACRO(...)

#define OCL_EMPTY
//==============================================================================


//---[ BOOLEAN STUFF ]----------------------------------------------------------
#define OCL_NOT_0 1
#define OCL_NOT_1 0

#define OCL_NOT4(X) OCL_NOT_##X
#define OCL_NOT3(X) OCL_NOT4(X)
#define OCL_NOT2(X) OCL_NOT3(X)
#define OCL_NOT(X)  OCL_NOT2(X)

#define OCL_AND_1_1 1
#define OCL_AND_1_0 0
#define OCL_AND_0_1 0
#define OCL_AND_0_0 0

#define OCL_AND4(X,Y) OCL_AND_##X##_##Y
#define OCL_AND3(X,Y) OCL_AND4(X,Y)
#define OCL_AND2(X,Y) OCL_AND3(X,Y)
#define OCL_AND(X,Y)  OCL_AND2(X,Y)

#define OCL_OR_1_1 1
#define OCL_OR_1_0 1
#define OCL_OR_0_1 1
#define OCL_OR_0_0 0

#define OCL_OR4(X,Y) OCL_OR_##X##_##Y
#define OCL_OR3(X,Y) OCL_OR4(X,Y)
#define OCL_OR2(X,Y) OCL_OR3(X,Y)
#define OCL_OR(X,Y)  OCL_OR2(X,Y)
//==============================================================================


//---[ MATH STUFF ]-------------------------------------------------------------
#define OCL_INC_0  1
#define OCL_INC_1  2
#define OCL_INC_2  3
#define OCL_INC_3  4
#define OCL_INC_4  5
#define OCL_INC_5  6
#define OCL_INC_6  7
#define OCL_INC_7  8
#define OCL_INC_8  9
#define OCL_INC_9  10
#define OCL_INC_10 11
#define OCL_INC_11 12
#define OCL_INC_12 13
#define OCL_INC_13 14
#define OCL_INC_14 15
#define OCL_INC_15 16
#define OCL_INC_16 17
#define OCL_INC_17 18
#define OCL_INC_18 19
#define OCL_INC_19 20
#define OCL_INC_20 21
#define OCL_INC_21 22
#define OCL_INC_22 23
#define OCL_INC_23 24
#define OCL_INC_24 25
#define OCL_INC_25 26

#define OCL_INC4(X) OCL_INC_##X
#define OCL_INC3(X) OCL_INC4(X)
#define OCL_INC2(X) OCL_INC3(X)
#define OCL_INC(X)  OCL_INC2(X)

#define OCL_DEC_0  0
#define OCL_DEC_1  0
#define OCL_DEC_2  1
#define OCL_DEC_3  2
#define OCL_DEC_4  3
#define OCL_DEC_5  4
#define OCL_DEC_6  5
#define OCL_DEC_7  6
#define OCL_DEC_8  7
#define OCL_DEC_9  8
#define OCL_DEC_10 9
#define OCL_DEC_11 10
#define OCL_DEC_12 11
#define OCL_DEC_13 12
#define OCL_DEC_14 13
#define OCL_DEC_15 14
#define OCL_DEC_16 15
#define OCL_DEC_17 16
#define OCL_DEC_18 17
#define OCL_DEC_19 18
#define OCL_DEC_20 19
#define OCL_DEC_21 20
#define OCL_DEC_22 21
#define OCL_DEC_23 22
#define OCL_DEC_24 23
#define OCL_DEC_25 24

#define OCL_DEC4(X) OCL_DEC_##X
#define OCL_DEC3(X) OCL_DEC4(X)
#define OCL_DEC2(X) OCL_DEC3(X)
#define OCL_DEC(X)  OCL_DEC2(X)

#define OCL_IF_0(T_EXPR, F_EXPR) F_EXPR
#define OCL_IF_1(T_EXPR, F_EXPR) T_EXPR

#define OCL_IF3(BOOL, T_EXPR, F_EXPR) OCL_IF_##BOOL(T_EXPR, F_EXPR)
#define OCL_IF2(BOOL, T_EXPR, F_EXPR) OCL_IF3(BOOL, T_EXPR, F_EXPR)
#define OCL_IF(BOOL, T_EXPR, F_EXPR)  OCL_IF2(BOOL, T_EXPR, F_EXPR)
//==============================================================================


//---[ ARITHMETIC OEPRATIONS ]--------------------------------------------------
//                   <  <= =  >= >  +  -  *  /

#define OCL_COMPARE_0_0 0, 1, 1, 1, 0, 0, 0, 0, 0
#define OCL_COMPARE_0_1 1, 1, 0, 0, 0, 1, -1, 0, 0
#define OCL_COMPARE_0_2 1, 1, 0, 0, 0, 2, -2, 0, 0
#define OCL_COMPARE_0_3 1, 1, 0, 0, 0, 3, -3, 0, 0
#define OCL_COMPARE_0_4 1, 1, 0, 0, 0, 4, -4, 0, 0
#define OCL_COMPARE_0_5 1, 1, 0, 0, 0, 5, -5, 0, 0
#define OCL_COMPARE_0_6 1, 1, 0, 0, 0, 6, -6, 0, 0
#define OCL_COMPARE_0_7 1, 1, 0, 0, 0, 7, -7, 0, 0
#define OCL_COMPARE_0_8 1, 1, 0, 0, 0, 8, -8, 0, 0
#define OCL_COMPARE_0_9 1, 1, 0, 0, 0, 9, -9, 0, 0
#define OCL_COMPARE_0_10 1, 1, 0, 0, 0, 10, -10, 0, 0
#define OCL_COMPARE_0_11 1, 1, 0, 0, 0, 11, -11, 0, 0
#define OCL_COMPARE_0_12 1, 1, 0, 0, 0, 12, -12, 0, 0
#define OCL_COMPARE_0_13 1, 1, 0, 0, 0, 13, -13, 0, 0
#define OCL_COMPARE_0_14 1, 1, 0, 0, 0, 14, -14, 0, 0
#define OCL_COMPARE_0_15 1, 1, 0, 0, 0, 15, -15, 0, 0
#define OCL_COMPARE_0_16 1, 1, 0, 0, 0, 16, -16, 0, 0
#define OCL_COMPARE_0_17 1, 1, 0, 0, 0, 17, -17, 0, 0
#define OCL_COMPARE_0_18 1, 1, 0, 0, 0, 18, -18, 0, 0
#define OCL_COMPARE_0_19 1, 1, 0, 0, 0, 19, -19, 0, 0
#define OCL_COMPARE_0_20 1, 1, 0, 0, 0, 20, -20, 0, 0
#define OCL_COMPARE_0_21 1, 1, 0, 0, 0, 21, -21, 0, 0
#define OCL_COMPARE_0_22 1, 1, 0, 0, 0, 22, -22, 0, 0
#define OCL_COMPARE_0_23 1, 1, 0, 0, 0, 23, -23, 0, 0
#define OCL_COMPARE_0_24 1, 1, 0, 0, 0, 24, -24, 0, 0
#define OCL_COMPARE_0_25 1, 1, 0, 0, 0, 25, -25, 0, 0

#define OCL_COMPARE_1_0 0, 0, 0, 1, 1, 1, 1, 0, 0
#define OCL_COMPARE_1_1 0, 1, 1, 1, 0, 2, 0, 1, 1
#define OCL_COMPARE_1_2 1, 1, 0, 0, 0, 3, -1, 2, 0
#define OCL_COMPARE_1_3 1, 1, 0, 0, 0, 4, -2, 3, 0
#define OCL_COMPARE_1_4 1, 1, 0, 0, 0, 5, -3, 4, 0
#define OCL_COMPARE_1_5 1, 1, 0, 0, 0, 6, -4, 5, 0
#define OCL_COMPARE_1_6 1, 1, 0, 0, 0, 7, -5, 6, 0
#define OCL_COMPARE_1_7 1, 1, 0, 0, 0, 8, -6, 7, 0
#define OCL_COMPARE_1_8 1, 1, 0, 0, 0, 9, -7, 8, 0
#define OCL_COMPARE_1_9 1, 1, 0, 0, 0, 10, -8, 9, 0
#define OCL_COMPARE_1_10 1, 1, 0, 0, 0, 11, -9, 10, 0
#define OCL_COMPARE_1_11 1, 1, 0, 0, 0, 12, -10, 11, 0
#define OCL_COMPARE_1_12 1, 1, 0, 0, 0, 13, -11, 12, 0
#define OCL_COMPARE_1_13 1, 1, 0, 0, 0, 14, -12, 13, 0
#define OCL_COMPARE_1_14 1, 1, 0, 0, 0, 15, -13, 14, 0
#define OCL_COMPARE_1_15 1, 1, 0, 0, 0, 16, -14, 15, 0
#define OCL_COMPARE_1_16 1, 1, 0, 0, 0, 17, -15, 16, 0
#define OCL_COMPARE_1_17 1, 1, 0, 0, 0, 18, -16, 17, 0
#define OCL_COMPARE_1_18 1, 1, 0, 0, 0, 19, -17, 18, 0
#define OCL_COMPARE_1_19 1, 1, 0, 0, 0, 20, -18, 19, 0
#define OCL_COMPARE_1_20 1, 1, 0, 0, 0, 21, -19, 20, 0
#define OCL_COMPARE_1_21 1, 1, 0, 0, 0, 22, -20, 21, 0
#define OCL_COMPARE_1_22 1, 1, 0, 0, 0, 23, -21, 22, 0
#define OCL_COMPARE_1_23 1, 1, 0, 0, 0, 24, -22, 23, 0
#define OCL_COMPARE_1_24 1, 1, 0, 0, 0, 25, -23, 24, 0
#define OCL_COMPARE_1_25 1, 1, 0, 0, 0, 26, -24, 25, 0

#define OCL_COMPARE_2_0 0, 0, 0, 1, 1, 2, 2, 0, 0
#define OCL_COMPARE_2_1 0, 0, 0, 1, 1, 3, 1, 2, 2
#define OCL_COMPARE_2_2 0, 1, 1, 1, 0, 4, 0, 4, 1
#define OCL_COMPARE_2_3 1, 1, 0, 0, 0, 5, -1, 6, 0
#define OCL_COMPARE_2_4 1, 1, 0, 0, 0, 6, -2, 8, 0
#define OCL_COMPARE_2_5 1, 1, 0, 0, 0, 7, -3, 10, 0
#define OCL_COMPARE_2_6 1, 1, 0, 0, 0, 8, -4, 12, 0
#define OCL_COMPARE_2_7 1, 1, 0, 0, 0, 9, -5, 14, 0
#define OCL_COMPARE_2_8 1, 1, 0, 0, 0, 10, -6, 16, 0
#define OCL_COMPARE_2_9 1, 1, 0, 0, 0, 11, -7, 18, 0
#define OCL_COMPARE_2_10 1, 1, 0, 0, 0, 12, -8, 20, 0
#define OCL_COMPARE_2_11 1, 1, 0, 0, 0, 13, -9, 22, 0
#define OCL_COMPARE_2_12 1, 1, 0, 0, 0, 14, -10, 24, 0
#define OCL_COMPARE_2_13 1, 1, 0, 0, 0, 15, -11, 26, 0
#define OCL_COMPARE_2_14 1, 1, 0, 0, 0, 16, -12, 28, 0
#define OCL_COMPARE_2_15 1, 1, 0, 0, 0, 17, -13, 30, 0
#define OCL_COMPARE_2_16 1, 1, 0, 0, 0, 18, -14, 32, 0
#define OCL_COMPARE_2_17 1, 1, 0, 0, 0, 19, -15, 34, 0
#define OCL_COMPARE_2_18 1, 1, 0, 0, 0, 20, -16, 36, 0
#define OCL_COMPARE_2_19 1, 1, 0, 0, 0, 21, -17, 38, 0
#define OCL_COMPARE_2_20 1, 1, 0, 0, 0, 22, -18, 40, 0
#define OCL_COMPARE_2_21 1, 1, 0, 0, 0, 23, -19, 42, 0
#define OCL_COMPARE_2_22 1, 1, 0, 0, 0, 24, -20, 44, 0
#define OCL_COMPARE_2_23 1, 1, 0, 0, 0, 25, -21, 46, 0
#define OCL_COMPARE_2_24 1, 1, 0, 0, 0, 26, -22, 48, 0
#define OCL_COMPARE_2_25 1, 1, 0, 0, 0, 27, -23, 50, 0

#define OCL_COMPARE_3_0 0, 0, 0, 1, 1, 3, 3, 0, 0
#define OCL_COMPARE_3_1 0, 0, 0, 1, 1, 4, 2, 3, 3
#define OCL_COMPARE_3_2 0, 0, 0, 1, 1, 5, 1, 6, 1
#define OCL_COMPARE_3_3 0, 1, 1, 1, 0, 6, 0, 9, 1
#define OCL_COMPARE_3_4 1, 1, 0, 0, 0, 7, -1, 12, 0
#define OCL_COMPARE_3_5 1, 1, 0, 0, 0, 8, -2, 15, 0
#define OCL_COMPARE_3_6 1, 1, 0, 0, 0, 9, -3, 18, 0
#define OCL_COMPARE_3_7 1, 1, 0, 0, 0, 10, -4, 21, 0
#define OCL_COMPARE_3_8 1, 1, 0, 0, 0, 11, -5, 24, 0
#define OCL_COMPARE_3_9 1, 1, 0, 0, 0, 12, -6, 27, 0
#define OCL_COMPARE_3_10 1, 1, 0, 0, 0, 13, -7, 30, 0
#define OCL_COMPARE_3_11 1, 1, 0, 0, 0, 14, -8, 33, 0
#define OCL_COMPARE_3_12 1, 1, 0, 0, 0, 15, -9, 36, 0
#define OCL_COMPARE_3_13 1, 1, 0, 0, 0, 16, -10, 39, 0
#define OCL_COMPARE_3_14 1, 1, 0, 0, 0, 17, -11, 42, 0
#define OCL_COMPARE_3_15 1, 1, 0, 0, 0, 18, -12, 45, 0
#define OCL_COMPARE_3_16 1, 1, 0, 0, 0, 19, -13, 48, 0
#define OCL_COMPARE_3_17 1, 1, 0, 0, 0, 20, -14, 51, 0
#define OCL_COMPARE_3_18 1, 1, 0, 0, 0, 21, -15, 54, 0
#define OCL_COMPARE_3_19 1, 1, 0, 0, 0, 22, -16, 57, 0
#define OCL_COMPARE_3_20 1, 1, 0, 0, 0, 23, -17, 60, 0
#define OCL_COMPARE_3_21 1, 1, 0, 0, 0, 24, -18, 63, 0
#define OCL_COMPARE_3_22 1, 1, 0, 0, 0, 25, -19, 66, 0
#define OCL_COMPARE_3_23 1, 1, 0, 0, 0, 26, -20, 69, 0
#define OCL_COMPARE_3_24 1, 1, 0, 0, 0, 27, -21, 72, 0
#define OCL_COMPARE_3_25 1, 1, 0, 0, 0, 28, -22, 75, 0

#define OCL_COMPARE_4_0 0, 0, 0, 1, 1, 4, 4, 0, 0
#define OCL_COMPARE_4_1 0, 0, 0, 1, 1, 5, 3, 4, 4
#define OCL_COMPARE_4_2 0, 0, 0, 1, 1, 6, 2, 8, 2
#define OCL_COMPARE_4_3 0, 0, 0, 1, 1, 7, 1, 12, 1
#define OCL_COMPARE_4_4 0, 1, 1, 1, 0, 8, 0, 16, 1
#define OCL_COMPARE_4_5 1, 1, 0, 0, 0, 9, -1, 20, 0
#define OCL_COMPARE_4_6 1, 1, 0, 0, 0, 10, -2, 24, 0
#define OCL_COMPARE_4_7 1, 1, 0, 0, 0, 11, -3, 28, 0
#define OCL_COMPARE_4_8 1, 1, 0, 0, 0, 12, -4, 32, 0
#define OCL_COMPARE_4_9 1, 1, 0, 0, 0, 13, -5, 36, 0
#define OCL_COMPARE_4_10 1, 1, 0, 0, 0, 14, -6, 40, 0
#define OCL_COMPARE_4_11 1, 1, 0, 0, 0, 15, -7, 44, 0
#define OCL_COMPARE_4_12 1, 1, 0, 0, 0, 16, -8, 48, 0
#define OCL_COMPARE_4_13 1, 1, 0, 0, 0, 17, -9, 52, 0
#define OCL_COMPARE_4_14 1, 1, 0, 0, 0, 18, -10, 56, 0
#define OCL_COMPARE_4_15 1, 1, 0, 0, 0, 19, -11, 60, 0
#define OCL_COMPARE_4_16 1, 1, 0, 0, 0, 20, -12, 64, 0
#define OCL_COMPARE_4_17 1, 1, 0, 0, 0, 21, -13, 68, 0
#define OCL_COMPARE_4_18 1, 1, 0, 0, 0, 22, -14, 72, 0
#define OCL_COMPARE_4_19 1, 1, 0, 0, 0, 23, -15, 76, 0
#define OCL_COMPARE_4_20 1, 1, 0, 0, 0, 24, -16, 80, 0
#define OCL_COMPARE_4_21 1, 1, 0, 0, 0, 25, -17, 84, 0
#define OCL_COMPARE_4_22 1, 1, 0, 0, 0, 26, -18, 88, 0
#define OCL_COMPARE_4_23 1, 1, 0, 0, 0, 27, -19, 92, 0
#define OCL_COMPARE_4_24 1, 1, 0, 0, 0, 28, -20, 96, 0
#define OCL_COMPARE_4_25 1, 1, 0, 0, 0, 29, -21, 100, 0

#define OCL_COMPARE_5_0 0, 0, 0, 1, 1, 5, 5, 0, 0
#define OCL_COMPARE_5_1 0, 0, 0, 1, 1, 6, 4, 5, 5
#define OCL_COMPARE_5_2 0, 0, 0, 1, 1, 7, 3, 10, 2
#define OCL_COMPARE_5_3 0, 0, 0, 1, 1, 8, 2, 15, 1
#define OCL_COMPARE_5_4 0, 0, 0, 1, 1, 9, 1, 20, 1
#define OCL_COMPARE_5_5 0, 1, 1, 1, 0, 10, 0, 25, 1
#define OCL_COMPARE_5_6 1, 1, 0, 0, 0, 11, -1, 30, 0
#define OCL_COMPARE_5_7 1, 1, 0, 0, 0, 12, -2, 35, 0
#define OCL_COMPARE_5_8 1, 1, 0, 0, 0, 13, -3, 40, 0
#define OCL_COMPARE_5_9 1, 1, 0, 0, 0, 14, -4, 45, 0
#define OCL_COMPARE_5_10 1, 1, 0, 0, 0, 15, -5, 50, 0
#define OCL_COMPARE_5_11 1, 1, 0, 0, 0, 16, -6, 55, 0
#define OCL_COMPARE_5_12 1, 1, 0, 0, 0, 17, -7, 60, 0
#define OCL_COMPARE_5_13 1, 1, 0, 0, 0, 18, -8, 65, 0
#define OCL_COMPARE_5_14 1, 1, 0, 0, 0, 19, -9, 70, 0
#define OCL_COMPARE_5_15 1, 1, 0, 0, 0, 20, -10, 75, 0
#define OCL_COMPARE_5_16 1, 1, 0, 0, 0, 21, -11, 80, 0
#define OCL_COMPARE_5_17 1, 1, 0, 0, 0, 22, -12, 85, 0
#define OCL_COMPARE_5_18 1, 1, 0, 0, 0, 23, -13, 90, 0
#define OCL_COMPARE_5_19 1, 1, 0, 0, 0, 24, -14, 95, 0
#define OCL_COMPARE_5_20 1, 1, 0, 0, 0, 25, -15, 100, 0
#define OCL_COMPARE_5_21 1, 1, 0, 0, 0, 26, -16, 105, 0
#define OCL_COMPARE_5_22 1, 1, 0, 0, 0, 27, -17, 110, 0
#define OCL_COMPARE_5_23 1, 1, 0, 0, 0, 28, -18, 115, 0
#define OCL_COMPARE_5_24 1, 1, 0, 0, 0, 29, -19, 120, 0
#define OCL_COMPARE_5_25 1, 1, 0, 0, 0, 30, -20, 125, 0

#define OCL_COMPARE_6_0 0, 0, 0, 1, 1, 6, 6, 0, 0
#define OCL_COMPARE_6_1 0, 0, 0, 1, 1, 7, 5, 6, 6
#define OCL_COMPARE_6_2 0, 0, 0, 1, 1, 8, 4, 12, 3
#define OCL_COMPARE_6_3 0, 0, 0, 1, 1, 9, 3, 18, 2
#define OCL_COMPARE_6_4 0, 0, 0, 1, 1, 10, 2, 24, 1
#define OCL_COMPARE_6_5 0, 0, 0, 1, 1, 11, 1, 30, 1
#define OCL_COMPARE_6_6 0, 1, 1, 1, 0, 12, 0, 36, 1
#define OCL_COMPARE_6_7 1, 1, 0, 0, 0, 13, -1, 42, 0
#define OCL_COMPARE_6_8 1, 1, 0, 0, 0, 14, -2, 48, 0
#define OCL_COMPARE_6_9 1, 1, 0, 0, 0, 15, -3, 54, 0
#define OCL_COMPARE_6_10 1, 1, 0, 0, 0, 16, -4, 60, 0
#define OCL_COMPARE_6_11 1, 1, 0, 0, 0, 17, -5, 66, 0
#define OCL_COMPARE_6_12 1, 1, 0, 0, 0, 18, -6, 72, 0
#define OCL_COMPARE_6_13 1, 1, 0, 0, 0, 19, -7, 78, 0
#define OCL_COMPARE_6_14 1, 1, 0, 0, 0, 20, -8, 84, 0
#define OCL_COMPARE_6_15 1, 1, 0, 0, 0, 21, -9, 90, 0
#define OCL_COMPARE_6_16 1, 1, 0, 0, 0, 22, -10, 96, 0
#define OCL_COMPARE_6_17 1, 1, 0, 0, 0, 23, -11, 102, 0
#define OCL_COMPARE_6_18 1, 1, 0, 0, 0, 24, -12, 108, 0
#define OCL_COMPARE_6_19 1, 1, 0, 0, 0, 25, -13, 114, 0
#define OCL_COMPARE_6_20 1, 1, 0, 0, 0, 26, -14, 120, 0
#define OCL_COMPARE_6_21 1, 1, 0, 0, 0, 27, -15, 126, 0
#define OCL_COMPARE_6_22 1, 1, 0, 0, 0, 28, -16, 132, 0
#define OCL_COMPARE_6_23 1, 1, 0, 0, 0, 29, -17, 138, 0
#define OCL_COMPARE_6_24 1, 1, 0, 0, 0, 30, -18, 144, 0
#define OCL_COMPARE_6_25 1, 1, 0, 0, 0, 31, -19, 150, 0

#define OCL_COMPARE_7_0 0, 0, 0, 1, 1, 7, 7, 0, 0
#define OCL_COMPARE_7_1 0, 0, 0, 1, 1, 8, 6, 7, 7
#define OCL_COMPARE_7_2 0, 0, 0, 1, 1, 9, 5, 14, 3
#define OCL_COMPARE_7_3 0, 0, 0, 1, 1, 10, 4, 21, 2
#define OCL_COMPARE_7_4 0, 0, 0, 1, 1, 11, 3, 28, 1
#define OCL_COMPARE_7_5 0, 0, 0, 1, 1, 12, 2, 35, 1
#define OCL_COMPARE_7_6 0, 0, 0, 1, 1, 13, 1, 42, 1
#define OCL_COMPARE_7_7 0, 1, 1, 1, 0, 14, 0, 49, 1
#define OCL_COMPARE_7_8 1, 1, 0, 0, 0, 15, -1, 56, 0
#define OCL_COMPARE_7_9 1, 1, 0, 0, 0, 16, -2, 63, 0
#define OCL_COMPARE_7_10 1, 1, 0, 0, 0, 17, -3, 70, 0
#define OCL_COMPARE_7_11 1, 1, 0, 0, 0, 18, -4, 77, 0
#define OCL_COMPARE_7_12 1, 1, 0, 0, 0, 19, -5, 84, 0
#define OCL_COMPARE_7_13 1, 1, 0, 0, 0, 20, -6, 91, 0
#define OCL_COMPARE_7_14 1, 1, 0, 0, 0, 21, -7, 98, 0
#define OCL_COMPARE_7_15 1, 1, 0, 0, 0, 22, -8, 105, 0
#define OCL_COMPARE_7_16 1, 1, 0, 0, 0, 23, -9, 112, 0
#define OCL_COMPARE_7_17 1, 1, 0, 0, 0, 24, -10, 119, 0
#define OCL_COMPARE_7_18 1, 1, 0, 0, 0, 25, -11, 126, 0
#define OCL_COMPARE_7_19 1, 1, 0, 0, 0, 26, -12, 133, 0
#define OCL_COMPARE_7_20 1, 1, 0, 0, 0, 27, -13, 140, 0
#define OCL_COMPARE_7_21 1, 1, 0, 0, 0, 28, -14, 147, 0
#define OCL_COMPARE_7_22 1, 1, 0, 0, 0, 29, -15, 154, 0
#define OCL_COMPARE_7_23 1, 1, 0, 0, 0, 30, -16, 161, 0
#define OCL_COMPARE_7_24 1, 1, 0, 0, 0, 31, -17, 168, 0
#define OCL_COMPARE_7_25 1, 1, 0, 0, 0, 32, -18, 175, 0

#define OCL_COMPARE_8_0 0, 0, 0, 1, 1, 8, 8, 0, 0
#define OCL_COMPARE_8_1 0, 0, 0, 1, 1, 9, 7, 8, 8
#define OCL_COMPARE_8_2 0, 0, 0, 1, 1, 10, 6, 16, 4
#define OCL_COMPARE_8_3 0, 0, 0, 1, 1, 11, 5, 24, 2
#define OCL_COMPARE_8_4 0, 0, 0, 1, 1, 12, 4, 32, 2
#define OCL_COMPARE_8_5 0, 0, 0, 1, 1, 13, 3, 40, 1
#define OCL_COMPARE_8_6 0, 0, 0, 1, 1, 14, 2, 48, 1
#define OCL_COMPARE_8_7 0, 0, 0, 1, 1, 15, 1, 56, 1
#define OCL_COMPARE_8_8 0, 1, 1, 1, 0, 16, 0, 64, 1
#define OCL_COMPARE_8_9 1, 1, 0, 0, 0, 17, -1, 72, 0
#define OCL_COMPARE_8_10 1, 1, 0, 0, 0, 18, -2, 80, 0
#define OCL_COMPARE_8_11 1, 1, 0, 0, 0, 19, -3, 88, 0
#define OCL_COMPARE_8_12 1, 1, 0, 0, 0, 20, -4, 96, 0
#define OCL_COMPARE_8_13 1, 1, 0, 0, 0, 21, -5, 104, 0
#define OCL_COMPARE_8_14 1, 1, 0, 0, 0, 22, -6, 112, 0
#define OCL_COMPARE_8_15 1, 1, 0, 0, 0, 23, -7, 120, 0
#define OCL_COMPARE_8_16 1, 1, 0, 0, 0, 24, -8, 128, 0
#define OCL_COMPARE_8_17 1, 1, 0, 0, 0, 25, -9, 136, 0
#define OCL_COMPARE_8_18 1, 1, 0, 0, 0, 26, -10, 144, 0
#define OCL_COMPARE_8_19 1, 1, 0, 0, 0, 27, -11, 152, 0
#define OCL_COMPARE_8_20 1, 1, 0, 0, 0, 28, -12, 160, 0
#define OCL_COMPARE_8_21 1, 1, 0, 0, 0, 29, -13, 168, 0
#define OCL_COMPARE_8_22 1, 1, 0, 0, 0, 30, -14, 176, 0
#define OCL_COMPARE_8_23 1, 1, 0, 0, 0, 31, -15, 184, 0
#define OCL_COMPARE_8_24 1, 1, 0, 0, 0, 32, -16, 192, 0
#define OCL_COMPARE_8_25 1, 1, 0, 0, 0, 33, -17, 200, 0

#define OCL_COMPARE_9_0 0, 0, 0, 1, 1, 9, 9, 0, 0
#define OCL_COMPARE_9_1 0, 0, 0, 1, 1, 10, 8, 9, 9
#define OCL_COMPARE_9_2 0, 0, 0, 1, 1, 11, 7, 18, 4
#define OCL_COMPARE_9_3 0, 0, 0, 1, 1, 12, 6, 27, 3
#define OCL_COMPARE_9_4 0, 0, 0, 1, 1, 13, 5, 36, 2
#define OCL_COMPARE_9_5 0, 0, 0, 1, 1, 14, 4, 45, 1
#define OCL_COMPARE_9_6 0, 0, 0, 1, 1, 15, 3, 54, 1
#define OCL_COMPARE_9_7 0, 0, 0, 1, 1, 16, 2, 63, 1
#define OCL_COMPARE_9_8 0, 0, 0, 1, 1, 17, 1, 72, 1
#define OCL_COMPARE_9_9 0, 1, 1, 1, 0, 18, 0, 81, 1
#define OCL_COMPARE_9_10 1, 1, 0, 0, 0, 19, -1, 90, 0
#define OCL_COMPARE_9_11 1, 1, 0, 0, 0, 20, -2, 99, 0
#define OCL_COMPARE_9_12 1, 1, 0, 0, 0, 21, -3, 108, 0
#define OCL_COMPARE_9_13 1, 1, 0, 0, 0, 22, -4, 117, 0
#define OCL_COMPARE_9_14 1, 1, 0, 0, 0, 23, -5, 126, 0
#define OCL_COMPARE_9_15 1, 1, 0, 0, 0, 24, -6, 135, 0
#define OCL_COMPARE_9_16 1, 1, 0, 0, 0, 25, -7, 144, 0
#define OCL_COMPARE_9_17 1, 1, 0, 0, 0, 26, -8, 153, 0
#define OCL_COMPARE_9_18 1, 1, 0, 0, 0, 27, -9, 162, 0
#define OCL_COMPARE_9_19 1, 1, 0, 0, 0, 28, -10, 171, 0
#define OCL_COMPARE_9_20 1, 1, 0, 0, 0, 29, -11, 180, 0
#define OCL_COMPARE_9_21 1, 1, 0, 0, 0, 30, -12, 189, 0
#define OCL_COMPARE_9_22 1, 1, 0, 0, 0, 31, -13, 198, 0
#define OCL_COMPARE_9_23 1, 1, 0, 0, 0, 32, -14, 207, 0
#define OCL_COMPARE_9_24 1, 1, 0, 0, 0, 33, -15, 216, 0
#define OCL_COMPARE_9_25 1, 1, 0, 0, 0, 34, -16, 225, 0

#define OCL_COMPARE_10_0 0, 0, 0, 1, 1, 10, 10, 0, 0
#define OCL_COMPARE_10_1 0, 0, 0, 1, 1, 11, 9, 10, 10
#define OCL_COMPARE_10_2 0, 0, 0, 1, 1, 12, 8, 20, 5
#define OCL_COMPARE_10_3 0, 0, 0, 1, 1, 13, 7, 30, 3
#define OCL_COMPARE_10_4 0, 0, 0, 1, 1, 14, 6, 40, 2
#define OCL_COMPARE_10_5 0, 0, 0, 1, 1, 15, 5, 50, 2
#define OCL_COMPARE_10_6 0, 0, 0, 1, 1, 16, 4, 60, 1
#define OCL_COMPARE_10_7 0, 0, 0, 1, 1, 17, 3, 70, 1
#define OCL_COMPARE_10_8 0, 0, 0, 1, 1, 18, 2, 80, 1
#define OCL_COMPARE_10_9 0, 0, 0, 1, 1, 19, 1, 90, 1
#define OCL_COMPARE_10_10 0, 1, 1, 1, 0, 20, 0, 100, 1
#define OCL_COMPARE_10_11 1, 1, 0, 0, 0, 21, -1, 110, 0
#define OCL_COMPARE_10_12 1, 1, 0, 0, 0, 22, -2, 120, 0
#define OCL_COMPARE_10_13 1, 1, 0, 0, 0, 23, -3, 130, 0
#define OCL_COMPARE_10_14 1, 1, 0, 0, 0, 24, -4, 140, 0
#define OCL_COMPARE_10_15 1, 1, 0, 0, 0, 25, -5, 150, 0
#define OCL_COMPARE_10_16 1, 1, 0, 0, 0, 26, -6, 160, 0
#define OCL_COMPARE_10_17 1, 1, 0, 0, 0, 27, -7, 170, 0
#define OCL_COMPARE_10_18 1, 1, 0, 0, 0, 28, -8, 180, 0
#define OCL_COMPARE_10_19 1, 1, 0, 0, 0, 29, -9, 190, 0
#define OCL_COMPARE_10_20 1, 1, 0, 0, 0, 30, -10, 200, 0
#define OCL_COMPARE_10_21 1, 1, 0, 0, 0, 31, -11, 210, 0
#define OCL_COMPARE_10_22 1, 1, 0, 0, 0, 32, -12, 220, 0
#define OCL_COMPARE_10_23 1, 1, 0, 0, 0, 33, -13, 230, 0
#define OCL_COMPARE_10_24 1, 1, 0, 0, 0, 34, -14, 240, 0
#define OCL_COMPARE_10_25 1, 1, 0, 0, 0, 35, -15, 250, 0

#define OCL_COMPARE_11_0 0, 0, 0, 1, 1, 11, 11, 0, 0
#define OCL_COMPARE_11_1 0, 0, 0, 1, 1, 12, 10, 11, 11
#define OCL_COMPARE_11_2 0, 0, 0, 1, 1, 13, 9, 22, 5
#define OCL_COMPARE_11_3 0, 0, 0, 1, 1, 14, 8, 33, 3
#define OCL_COMPARE_11_4 0, 0, 0, 1, 1, 15, 7, 44, 2
#define OCL_COMPARE_11_5 0, 0, 0, 1, 1, 16, 6, 55, 2
#define OCL_COMPARE_11_6 0, 0, 0, 1, 1, 17, 5, 66, 1
#define OCL_COMPARE_11_7 0, 0, 0, 1, 1, 18, 4, 77, 1
#define OCL_COMPARE_11_8 0, 0, 0, 1, 1, 19, 3, 88, 1
#define OCL_COMPARE_11_9 0, 0, 0, 1, 1, 20, 2, 99, 1
#define OCL_COMPARE_11_10 0, 0, 0, 1, 1, 21, 1, 110, 1
#define OCL_COMPARE_11_11 0, 1, 1, 1, 0, 22, 0, 121, 1
#define OCL_COMPARE_11_12 1, 1, 0, 0, 0, 23, -1, 132, 0
#define OCL_COMPARE_11_13 1, 1, 0, 0, 0, 24, -2, 143, 0
#define OCL_COMPARE_11_14 1, 1, 0, 0, 0, 25, -3, 154, 0
#define OCL_COMPARE_11_15 1, 1, 0, 0, 0, 26, -4, 165, 0
#define OCL_COMPARE_11_16 1, 1, 0, 0, 0, 27, -5, 176, 0
#define OCL_COMPARE_11_17 1, 1, 0, 0, 0, 28, -6, 187, 0
#define OCL_COMPARE_11_18 1, 1, 0, 0, 0, 29, -7, 198, 0
#define OCL_COMPARE_11_19 1, 1, 0, 0, 0, 30, -8, 209, 0
#define OCL_COMPARE_11_20 1, 1, 0, 0, 0, 31, -9, 220, 0
#define OCL_COMPARE_11_21 1, 1, 0, 0, 0, 32, -10, 231, 0
#define OCL_COMPARE_11_22 1, 1, 0, 0, 0, 33, -11, 242, 0
#define OCL_COMPARE_11_23 1, 1, 0, 0, 0, 34, -12, 253, 0
#define OCL_COMPARE_11_24 1, 1, 0, 0, 0, 35, -13, 264, 0
#define OCL_COMPARE_11_25 1, 1, 0, 0, 0, 36, -14, 275, 0

#define OCL_COMPARE_12_0 0, 0, 0, 1, 1, 12, 12, 0, 0
#define OCL_COMPARE_12_1 0, 0, 0, 1, 1, 13, 11, 12, 12
#define OCL_COMPARE_12_2 0, 0, 0, 1, 1, 14, 10, 24, 6
#define OCL_COMPARE_12_3 0, 0, 0, 1, 1, 15, 9, 36, 4
#define OCL_COMPARE_12_4 0, 0, 0, 1, 1, 16, 8, 48, 3
#define OCL_COMPARE_12_5 0, 0, 0, 1, 1, 17, 7, 60, 2
#define OCL_COMPARE_12_6 0, 0, 0, 1, 1, 18, 6, 72, 2
#define OCL_COMPARE_12_7 0, 0, 0, 1, 1, 19, 5, 84, 1
#define OCL_COMPARE_12_8 0, 0, 0, 1, 1, 20, 4, 96, 1
#define OCL_COMPARE_12_9 0, 0, 0, 1, 1, 21, 3, 108, 1
#define OCL_COMPARE_12_10 0, 0, 0, 1, 1, 22, 2, 120, 1
#define OCL_COMPARE_12_11 0, 0, 0, 1, 1, 23, 1, 132, 1
#define OCL_COMPARE_12_12 0, 1, 1, 1, 0, 24, 0, 144, 1
#define OCL_COMPARE_12_13 1, 1, 0, 0, 0, 25, -1, 156, 0
#define OCL_COMPARE_12_14 1, 1, 0, 0, 0, 26, -2, 168, 0
#define OCL_COMPARE_12_15 1, 1, 0, 0, 0, 27, -3, 180, 0
#define OCL_COMPARE_12_16 1, 1, 0, 0, 0, 28, -4, 192, 0
#define OCL_COMPARE_12_17 1, 1, 0, 0, 0, 29, -5, 204, 0
#define OCL_COMPARE_12_18 1, 1, 0, 0, 0, 30, -6, 216, 0
#define OCL_COMPARE_12_19 1, 1, 0, 0, 0, 31, -7, 228, 0
#define OCL_COMPARE_12_20 1, 1, 0, 0, 0, 32, -8, 240, 0
#define OCL_COMPARE_12_21 1, 1, 0, 0, 0, 33, -9, 252, 0
#define OCL_COMPARE_12_22 1, 1, 0, 0, 0, 34, -10, 264, 0
#define OCL_COMPARE_12_23 1, 1, 0, 0, 0, 35, -11, 276, 0
#define OCL_COMPARE_12_24 1, 1, 0, 0, 0, 36, -12, 288, 0
#define OCL_COMPARE_12_25 1, 1, 0, 0, 0, 37, -13, 300, 0

#define OCL_COMPARE_13_0 0, 0, 0, 1, 1, 13, 13, 0, 0
#define OCL_COMPARE_13_1 0, 0, 0, 1, 1, 14, 12, 13, 13
#define OCL_COMPARE_13_2 0, 0, 0, 1, 1, 15, 11, 26, 6
#define OCL_COMPARE_13_3 0, 0, 0, 1, 1, 16, 10, 39, 4
#define OCL_COMPARE_13_4 0, 0, 0, 1, 1, 17, 9, 52, 3
#define OCL_COMPARE_13_5 0, 0, 0, 1, 1, 18, 8, 65, 2
#define OCL_COMPARE_13_6 0, 0, 0, 1, 1, 19, 7, 78, 2
#define OCL_COMPARE_13_7 0, 0, 0, 1, 1, 20, 6, 91, 1
#define OCL_COMPARE_13_8 0, 0, 0, 1, 1, 21, 5, 104, 1
#define OCL_COMPARE_13_9 0, 0, 0, 1, 1, 22, 4, 117, 1
#define OCL_COMPARE_13_10 0, 0, 0, 1, 1, 23, 3, 130, 1
#define OCL_COMPARE_13_11 0, 0, 0, 1, 1, 24, 2, 143, 1
#define OCL_COMPARE_13_12 0, 0, 0, 1, 1, 25, 1, 156, 1
#define OCL_COMPARE_13_13 0, 1, 1, 1, 0, 26, 0, 169, 1
#define OCL_COMPARE_13_14 1, 1, 0, 0, 0, 27, -1, 182, 0
#define OCL_COMPARE_13_15 1, 1, 0, 0, 0, 28, -2, 195, 0
#define OCL_COMPARE_13_16 1, 1, 0, 0, 0, 29, -3, 208, 0
#define OCL_COMPARE_13_17 1, 1, 0, 0, 0, 30, -4, 221, 0
#define OCL_COMPARE_13_18 1, 1, 0, 0, 0, 31, -5, 234, 0
#define OCL_COMPARE_13_19 1, 1, 0, 0, 0, 32, -6, 247, 0
#define OCL_COMPARE_13_20 1, 1, 0, 0, 0, 33, -7, 260, 0
#define OCL_COMPARE_13_21 1, 1, 0, 0, 0, 34, -8, 273, 0
#define OCL_COMPARE_13_22 1, 1, 0, 0, 0, 35, -9, 286, 0
#define OCL_COMPARE_13_23 1, 1, 0, 0, 0, 36, -10, 299, 0
#define OCL_COMPARE_13_24 1, 1, 0, 0, 0, 37, -11, 312, 0
#define OCL_COMPARE_13_25 1, 1, 0, 0, 0, 38, -12, 325, 0

#define OCL_COMPARE_14_0 0, 0, 0, 1, 1, 14, 14, 0, 0
#define OCL_COMPARE_14_1 0, 0, 0, 1, 1, 15, 13, 14, 14
#define OCL_COMPARE_14_2 0, 0, 0, 1, 1, 16, 12, 28, 7
#define OCL_COMPARE_14_3 0, 0, 0, 1, 1, 17, 11, 42, 4
#define OCL_COMPARE_14_4 0, 0, 0, 1, 1, 18, 10, 56, 3
#define OCL_COMPARE_14_5 0, 0, 0, 1, 1, 19, 9, 70, 2
#define OCL_COMPARE_14_6 0, 0, 0, 1, 1, 20, 8, 84, 2
#define OCL_COMPARE_14_7 0, 0, 0, 1, 1, 21, 7, 98, 2
#define OCL_COMPARE_14_8 0, 0, 0, 1, 1, 22, 6, 112, 1
#define OCL_COMPARE_14_9 0, 0, 0, 1, 1, 23, 5, 126, 1
#define OCL_COMPARE_14_10 0, 0, 0, 1, 1, 24, 4, 140, 1
#define OCL_COMPARE_14_11 0, 0, 0, 1, 1, 25, 3, 154, 1
#define OCL_COMPARE_14_12 0, 0, 0, 1, 1, 26, 2, 168, 1
#define OCL_COMPARE_14_13 0, 0, 0, 1, 1, 27, 1, 182, 1
#define OCL_COMPARE_14_14 0, 1, 1, 1, 0, 28, 0, 196, 1
#define OCL_COMPARE_14_15 1, 1, 0, 0, 0, 29, -1, 210, 0
#define OCL_COMPARE_14_16 1, 1, 0, 0, 0, 30, -2, 224, 0
#define OCL_COMPARE_14_17 1, 1, 0, 0, 0, 31, -3, 238, 0
#define OCL_COMPARE_14_18 1, 1, 0, 0, 0, 32, -4, 252, 0
#define OCL_COMPARE_14_19 1, 1, 0, 0, 0, 33, -5, 266, 0
#define OCL_COMPARE_14_20 1, 1, 0, 0, 0, 34, -6, 280, 0
#define OCL_COMPARE_14_21 1, 1, 0, 0, 0, 35, -7, 294, 0
#define OCL_COMPARE_14_22 1, 1, 0, 0, 0, 36, -8, 308, 0
#define OCL_COMPARE_14_23 1, 1, 0, 0, 0, 37, -9, 322, 0
#define OCL_COMPARE_14_24 1, 1, 0, 0, 0, 38, -10, 336, 0
#define OCL_COMPARE_14_25 1, 1, 0, 0, 0, 39, -11, 350, 0

#define OCL_COMPARE_15_0 0, 0, 0, 1, 1, 15, 15, 0, 0
#define OCL_COMPARE_15_1 0, 0, 0, 1, 1, 16, 14, 15, 15
#define OCL_COMPARE_15_2 0, 0, 0, 1, 1, 17, 13, 30, 7
#define OCL_COMPARE_15_3 0, 0, 0, 1, 1, 18, 12, 45, 5
#define OCL_COMPARE_15_4 0, 0, 0, 1, 1, 19, 11, 60, 3
#define OCL_COMPARE_15_5 0, 0, 0, 1, 1, 20, 10, 75, 3
#define OCL_COMPARE_15_6 0, 0, 0, 1, 1, 21, 9, 90, 2
#define OCL_COMPARE_15_7 0, 0, 0, 1, 1, 22, 8, 105, 2
#define OCL_COMPARE_15_8 0, 0, 0, 1, 1, 23, 7, 120, 1
#define OCL_COMPARE_15_9 0, 0, 0, 1, 1, 24, 6, 135, 1
#define OCL_COMPARE_15_10 0, 0, 0, 1, 1, 25, 5, 150, 1
#define OCL_COMPARE_15_11 0, 0, 0, 1, 1, 26, 4, 165, 1
#define OCL_COMPARE_15_12 0, 0, 0, 1, 1, 27, 3, 180, 1
#define OCL_COMPARE_15_13 0, 0, 0, 1, 1, 28, 2, 195, 1
#define OCL_COMPARE_15_14 0, 0, 0, 1, 1, 29, 1, 210, 1
#define OCL_COMPARE_15_15 0, 1, 1, 1, 0, 30, 0, 225, 1
#define OCL_COMPARE_15_16 1, 1, 0, 0, 0, 31, -1, 240, 0
#define OCL_COMPARE_15_17 1, 1, 0, 0, 0, 32, -2, 255, 0
#define OCL_COMPARE_15_18 1, 1, 0, 0, 0, 33, -3, 270, 0
#define OCL_COMPARE_15_19 1, 1, 0, 0, 0, 34, -4, 285, 0
#define OCL_COMPARE_15_20 1, 1, 0, 0, 0, 35, -5, 300, 0
#define OCL_COMPARE_15_21 1, 1, 0, 0, 0, 36, -6, 315, 0
#define OCL_COMPARE_15_22 1, 1, 0, 0, 0, 37, -7, 330, 0
#define OCL_COMPARE_15_23 1, 1, 0, 0, 0, 38, -8, 345, 0
#define OCL_COMPARE_15_24 1, 1, 0, 0, 0, 39, -9, 360, 0
#define OCL_COMPARE_15_25 1, 1, 0, 0, 0, 40, -10, 375, 0

#define OCL_COMPARE_16_0 0, 0, 0, 1, 1, 16, 16, 0, 0
#define OCL_COMPARE_16_1 0, 0, 0, 1, 1, 17, 15, 16, 16
#define OCL_COMPARE_16_2 0, 0, 0, 1, 1, 18, 14, 32, 8
#define OCL_COMPARE_16_3 0, 0, 0, 1, 1, 19, 13, 48, 5
#define OCL_COMPARE_16_4 0, 0, 0, 1, 1, 20, 12, 64, 4
#define OCL_COMPARE_16_5 0, 0, 0, 1, 1, 21, 11, 80, 3
#define OCL_COMPARE_16_6 0, 0, 0, 1, 1, 22, 10, 96, 2
#define OCL_COMPARE_16_7 0, 0, 0, 1, 1, 23, 9, 112, 2
#define OCL_COMPARE_16_8 0, 0, 0, 1, 1, 24, 8, 128, 2
#define OCL_COMPARE_16_9 0, 0, 0, 1, 1, 25, 7, 144, 1
#define OCL_COMPARE_16_10 0, 0, 0, 1, 1, 26, 6, 160, 1
#define OCL_COMPARE_16_11 0, 0, 0, 1, 1, 27, 5, 176, 1
#define OCL_COMPARE_16_12 0, 0, 0, 1, 1, 28, 4, 192, 1
#define OCL_COMPARE_16_13 0, 0, 0, 1, 1, 29, 3, 208, 1
#define OCL_COMPARE_16_14 0, 0, 0, 1, 1, 30, 2, 224, 1
#define OCL_COMPARE_16_15 0, 0, 0, 1, 1, 31, 1, 240, 1
#define OCL_COMPARE_16_16 0, 1, 1, 1, 0, 32, 0, 256, 1
#define OCL_COMPARE_16_17 1, 1, 0, 0, 0, 33, -1, 272, 0
#define OCL_COMPARE_16_18 1, 1, 0, 0, 0, 34, -2, 288, 0
#define OCL_COMPARE_16_19 1, 1, 0, 0, 0, 35, -3, 304, 0
#define OCL_COMPARE_16_20 1, 1, 0, 0, 0, 36, -4, 320, 0
#define OCL_COMPARE_16_21 1, 1, 0, 0, 0, 37, -5, 336, 0
#define OCL_COMPARE_16_22 1, 1, 0, 0, 0, 38, -6, 352, 0
#define OCL_COMPARE_16_23 1, 1, 0, 0, 0, 39, -7, 368, 0
#define OCL_COMPARE_16_24 1, 1, 0, 0, 0, 40, -8, 384, 0
#define OCL_COMPARE_16_25 1, 1, 0, 0, 0, 41, -9, 400, 0

#define OCL_COMPARE_17_0 0, 0, 0, 1, 1, 17, 17, 0, 0
#define OCL_COMPARE_17_1 0, 0, 0, 1, 1, 18, 16, 17, 17
#define OCL_COMPARE_17_2 0, 0, 0, 1, 1, 19, 15, 34, 8
#define OCL_COMPARE_17_3 0, 0, 0, 1, 1, 20, 14, 51, 5
#define OCL_COMPARE_17_4 0, 0, 0, 1, 1, 21, 13, 68, 4
#define OCL_COMPARE_17_5 0, 0, 0, 1, 1, 22, 12, 85, 3
#define OCL_COMPARE_17_6 0, 0, 0, 1, 1, 23, 11, 102, 2
#define OCL_COMPARE_17_7 0, 0, 0, 1, 1, 24, 10, 119, 2
#define OCL_COMPARE_17_8 0, 0, 0, 1, 1, 25, 9, 136, 2
#define OCL_COMPARE_17_9 0, 0, 0, 1, 1, 26, 8, 153, 1
#define OCL_COMPARE_17_10 0, 0, 0, 1, 1, 27, 7, 170, 1
#define OCL_COMPARE_17_11 0, 0, 0, 1, 1, 28, 6, 187, 1
#define OCL_COMPARE_17_12 0, 0, 0, 1, 1, 29, 5, 204, 1
#define OCL_COMPARE_17_13 0, 0, 0, 1, 1, 30, 4, 221, 1
#define OCL_COMPARE_17_14 0, 0, 0, 1, 1, 31, 3, 238, 1
#define OCL_COMPARE_17_15 0, 0, 0, 1, 1, 32, 2, 255, 1
#define OCL_COMPARE_17_16 0, 0, 0, 1, 1, 33, 1, 272, 1
#define OCL_COMPARE_17_17 0, 1, 1, 1, 0, 34, 0, 289, 1
#define OCL_COMPARE_17_18 1, 1, 0, 0, 0, 35, -1, 306, 0
#define OCL_COMPARE_17_19 1, 1, 0, 0, 0, 36, -2, 323, 0
#define OCL_COMPARE_17_20 1, 1, 0, 0, 0, 37, -3, 340, 0
#define OCL_COMPARE_17_21 1, 1, 0, 0, 0, 38, -4, 357, 0
#define OCL_COMPARE_17_22 1, 1, 0, 0, 0, 39, -5, 374, 0
#define OCL_COMPARE_17_23 1, 1, 0, 0, 0, 40, -6, 391, 0
#define OCL_COMPARE_17_24 1, 1, 0, 0, 0, 41, -7, 408, 0
#define OCL_COMPARE_17_25 1, 1, 0, 0, 0, 42, -8, 425, 0

#define OCL_COMPARE_18_0 0, 0, 0, 1, 1, 18, 18, 0, 0
#define OCL_COMPARE_18_1 0, 0, 0, 1, 1, 19, 17, 18, 18
#define OCL_COMPARE_18_2 0, 0, 0, 1, 1, 20, 16, 36, 9
#define OCL_COMPARE_18_3 0, 0, 0, 1, 1, 21, 15, 54, 6
#define OCL_COMPARE_18_4 0, 0, 0, 1, 1, 22, 14, 72, 4
#define OCL_COMPARE_18_5 0, 0, 0, 1, 1, 23, 13, 90, 3
#define OCL_COMPARE_18_6 0, 0, 0, 1, 1, 24, 12, 108, 3
#define OCL_COMPARE_18_7 0, 0, 0, 1, 1, 25, 11, 126, 2
#define OCL_COMPARE_18_8 0, 0, 0, 1, 1, 26, 10, 144, 2
#define OCL_COMPARE_18_9 0, 0, 0, 1, 1, 27, 9, 162, 2
#define OCL_COMPARE_18_10 0, 0, 0, 1, 1, 28, 8, 180, 1
#define OCL_COMPARE_18_11 0, 0, 0, 1, 1, 29, 7, 198, 1
#define OCL_COMPARE_18_12 0, 0, 0, 1, 1, 30, 6, 216, 1
#define OCL_COMPARE_18_13 0, 0, 0, 1, 1, 31, 5, 234, 1
#define OCL_COMPARE_18_14 0, 0, 0, 1, 1, 32, 4, 252, 1
#define OCL_COMPARE_18_15 0, 0, 0, 1, 1, 33, 3, 270, 1
#define OCL_COMPARE_18_16 0, 0, 0, 1, 1, 34, 2, 288, 1
#define OCL_COMPARE_18_17 0, 0, 0, 1, 1, 35, 1, 306, 1
#define OCL_COMPARE_18_18 0, 1, 1, 1, 0, 36, 0, 324, 1
#define OCL_COMPARE_18_19 1, 1, 0, 0, 0, 37, -1, 342, 0
#define OCL_COMPARE_18_20 1, 1, 0, 0, 0, 38, -2, 360, 0
#define OCL_COMPARE_18_21 1, 1, 0, 0, 0, 39, -3, 378, 0
#define OCL_COMPARE_18_22 1, 1, 0, 0, 0, 40, -4, 396, 0
#define OCL_COMPARE_18_23 1, 1, 0, 0, 0, 41, -5, 414, 0
#define OCL_COMPARE_18_24 1, 1, 0, 0, 0, 42, -6, 432, 0
#define OCL_COMPARE_18_25 1, 1, 0, 0, 0, 43, -7, 450, 0

#define OCL_COMPARE_19_0 0, 0, 0, 1, 1, 19, 19, 0, 0
#define OCL_COMPARE_19_1 0, 0, 0, 1, 1, 20, 18, 19, 19
#define OCL_COMPARE_19_2 0, 0, 0, 1, 1, 21, 17, 38, 9
#define OCL_COMPARE_19_3 0, 0, 0, 1, 1, 22, 16, 57, 6
#define OCL_COMPARE_19_4 0, 0, 0, 1, 1, 23, 15, 76, 4
#define OCL_COMPARE_19_5 0, 0, 0, 1, 1, 24, 14, 95, 3
#define OCL_COMPARE_19_6 0, 0, 0, 1, 1, 25, 13, 114, 3
#define OCL_COMPARE_19_7 0, 0, 0, 1, 1, 26, 12, 133, 2
#define OCL_COMPARE_19_8 0, 0, 0, 1, 1, 27, 11, 152, 2
#define OCL_COMPARE_19_9 0, 0, 0, 1, 1, 28, 10, 171, 2
#define OCL_COMPARE_19_10 0, 0, 0, 1, 1, 29, 9, 190, 1
#define OCL_COMPARE_19_11 0, 0, 0, 1, 1, 30, 8, 209, 1
#define OCL_COMPARE_19_12 0, 0, 0, 1, 1, 31, 7, 228, 1
#define OCL_COMPARE_19_13 0, 0, 0, 1, 1, 32, 6, 247, 1
#define OCL_COMPARE_19_14 0, 0, 0, 1, 1, 33, 5, 266, 1
#define OCL_COMPARE_19_15 0, 0, 0, 1, 1, 34, 4, 285, 1
#define OCL_COMPARE_19_16 0, 0, 0, 1, 1, 35, 3, 304, 1
#define OCL_COMPARE_19_17 0, 0, 0, 1, 1, 36, 2, 323, 1
#define OCL_COMPARE_19_18 0, 0, 0, 1, 1, 37, 1, 342, 1
#define OCL_COMPARE_19_19 0, 1, 1, 1, 0, 38, 0, 361, 1
#define OCL_COMPARE_19_20 1, 1, 0, 0, 0, 39, -1, 380, 0
#define OCL_COMPARE_19_21 1, 1, 0, 0, 0, 40, -2, 399, 0
#define OCL_COMPARE_19_22 1, 1, 0, 0, 0, 41, -3, 418, 0
#define OCL_COMPARE_19_23 1, 1, 0, 0, 0, 42, -4, 437, 0
#define OCL_COMPARE_19_24 1, 1, 0, 0, 0, 43, -5, 456, 0
#define OCL_COMPARE_19_25 1, 1, 0, 0, 0, 44, -6, 475, 0

#define OCL_COMPARE_20_0 0, 0, 0, 1, 1, 20, 20, 0, 0
#define OCL_COMPARE_20_1 0, 0, 0, 1, 1, 21, 19, 20, 20
#define OCL_COMPARE_20_2 0, 0, 0, 1, 1, 22, 18, 40, 10
#define OCL_COMPARE_20_3 0, 0, 0, 1, 1, 23, 17, 60, 6
#define OCL_COMPARE_20_4 0, 0, 0, 1, 1, 24, 16, 80, 5
#define OCL_COMPARE_20_5 0, 0, 0, 1, 1, 25, 15, 100, 4
#define OCL_COMPARE_20_6 0, 0, 0, 1, 1, 26, 14, 120, 3
#define OCL_COMPARE_20_7 0, 0, 0, 1, 1, 27, 13, 140, 2
#define OCL_COMPARE_20_8 0, 0, 0, 1, 1, 28, 12, 160, 2
#define OCL_COMPARE_20_9 0, 0, 0, 1, 1, 29, 11, 180, 2
#define OCL_COMPARE_20_10 0, 0, 0, 1, 1, 30, 10, 200, 2
#define OCL_COMPARE_20_11 0, 0, 0, 1, 1, 31, 9, 220, 1
#define OCL_COMPARE_20_12 0, 0, 0, 1, 1, 32, 8, 240, 1
#define OCL_COMPARE_20_13 0, 0, 0, 1, 1, 33, 7, 260, 1
#define OCL_COMPARE_20_14 0, 0, 0, 1, 1, 34, 6, 280, 1
#define OCL_COMPARE_20_15 0, 0, 0, 1, 1, 35, 5, 300, 1
#define OCL_COMPARE_20_16 0, 0, 0, 1, 1, 36, 4, 320, 1
#define OCL_COMPARE_20_17 0, 0, 0, 1, 1, 37, 3, 340, 1
#define OCL_COMPARE_20_18 0, 0, 0, 1, 1, 38, 2, 360, 1
#define OCL_COMPARE_20_19 0, 0, 0, 1, 1, 39, 1, 380, 1
#define OCL_COMPARE_20_20 0, 1, 1, 1, 0, 40, 0, 400, 1
#define OCL_COMPARE_20_21 1, 1, 0, 0, 0, 41, -1, 420, 0
#define OCL_COMPARE_20_22 1, 1, 0, 0, 0, 42, -2, 440, 0
#define OCL_COMPARE_20_23 1, 1, 0, 0, 0, 43, -3, 460, 0
#define OCL_COMPARE_20_24 1, 1, 0, 0, 0, 44, -4, 480, 0
#define OCL_COMPARE_20_25 1, 1, 0, 0, 0, 45, -5, 500, 0

#define OCL_COMPARE_21_0 0, 0, 0, 1, 1, 21, 21, 0, 0
#define OCL_COMPARE_21_1 0, 0, 0, 1, 1, 22, 20, 21, 21
#define OCL_COMPARE_21_2 0, 0, 0, 1, 1, 23, 19, 42, 10
#define OCL_COMPARE_21_3 0, 0, 0, 1, 1, 24, 18, 63, 7
#define OCL_COMPARE_21_4 0, 0, 0, 1, 1, 25, 17, 84, 5
#define OCL_COMPARE_21_5 0, 0, 0, 1, 1, 26, 16, 105, 4
#define OCL_COMPARE_21_6 0, 0, 0, 1, 1, 27, 15, 126, 3
#define OCL_COMPARE_21_7 0, 0, 0, 1, 1, 28, 14, 147, 3
#define OCL_COMPARE_21_8 0, 0, 0, 1, 1, 29, 13, 168, 2
#define OCL_COMPARE_21_9 0, 0, 0, 1, 1, 30, 12, 189, 2
#define OCL_COMPARE_21_10 0, 0, 0, 1, 1, 31, 11, 210, 2
#define OCL_COMPARE_21_11 0, 0, 0, 1, 1, 32, 10, 231, 1
#define OCL_COMPARE_21_12 0, 0, 0, 1, 1, 33, 9, 252, 1
#define OCL_COMPARE_21_13 0, 0, 0, 1, 1, 34, 8, 273, 1
#define OCL_COMPARE_21_14 0, 0, 0, 1, 1, 35, 7, 294, 1
#define OCL_COMPARE_21_15 0, 0, 0, 1, 1, 36, 6, 315, 1
#define OCL_COMPARE_21_16 0, 0, 0, 1, 1, 37, 5, 336, 1
#define OCL_COMPARE_21_17 0, 0, 0, 1, 1, 38, 4, 357, 1
#define OCL_COMPARE_21_18 0, 0, 0, 1, 1, 39, 3, 378, 1
#define OCL_COMPARE_21_19 0, 0, 0, 1, 1, 40, 2, 399, 1
#define OCL_COMPARE_21_20 0, 0, 0, 1, 1, 41, 1, 420, 1
#define OCL_COMPARE_21_21 0, 1, 1, 1, 0, 42, 0, 441, 1
#define OCL_COMPARE_21_22 1, 1, 0, 0, 0, 43, -1, 462, 0
#define OCL_COMPARE_21_23 1, 1, 0, 0, 0, 44, -2, 483, 0
#define OCL_COMPARE_21_24 1, 1, 0, 0, 0, 45, -3, 504, 0
#define OCL_COMPARE_21_25 1, 1, 0, 0, 0, 46, -4, 525, 0

#define OCL_COMPARE_22_0 0, 0, 0, 1, 1, 22, 22, 0, 0
#define OCL_COMPARE_22_1 0, 0, 0, 1, 1, 23, 21, 22, 22
#define OCL_COMPARE_22_2 0, 0, 0, 1, 1, 24, 20, 44, 11
#define OCL_COMPARE_22_3 0, 0, 0, 1, 1, 25, 19, 66, 7
#define OCL_COMPARE_22_4 0, 0, 0, 1, 1, 26, 18, 88, 5
#define OCL_COMPARE_22_5 0, 0, 0, 1, 1, 27, 17, 110, 4
#define OCL_COMPARE_22_6 0, 0, 0, 1, 1, 28, 16, 132, 3
#define OCL_COMPARE_22_7 0, 0, 0, 1, 1, 29, 15, 154, 3
#define OCL_COMPARE_22_8 0, 0, 0, 1, 1, 30, 14, 176, 2
#define OCL_COMPARE_22_9 0, 0, 0, 1, 1, 31, 13, 198, 2
#define OCL_COMPARE_22_10 0, 0, 0, 1, 1, 32, 12, 220, 2
#define OCL_COMPARE_22_11 0, 0, 0, 1, 1, 33, 11, 242, 2
#define OCL_COMPARE_22_12 0, 0, 0, 1, 1, 34, 10, 264, 1
#define OCL_COMPARE_22_13 0, 0, 0, 1, 1, 35, 9, 286, 1
#define OCL_COMPARE_22_14 0, 0, 0, 1, 1, 36, 8, 308, 1
#define OCL_COMPARE_22_15 0, 0, 0, 1, 1, 37, 7, 330, 1
#define OCL_COMPARE_22_16 0, 0, 0, 1, 1, 38, 6, 352, 1
#define OCL_COMPARE_22_17 0, 0, 0, 1, 1, 39, 5, 374, 1
#define OCL_COMPARE_22_18 0, 0, 0, 1, 1, 40, 4, 396, 1
#define OCL_COMPARE_22_19 0, 0, 0, 1, 1, 41, 3, 418, 1
#define OCL_COMPARE_22_20 0, 0, 0, 1, 1, 42, 2, 440, 1
#define OCL_COMPARE_22_21 0, 0, 0, 1, 1, 43, 1, 462, 1
#define OCL_COMPARE_22_22 0, 1, 1, 1, 0, 44, 0, 484, 1
#define OCL_COMPARE_22_23 1, 1, 0, 0, 0, 45, -1, 506, 0
#define OCL_COMPARE_22_24 1, 1, 0, 0, 0, 46, -2, 528, 0
#define OCL_COMPARE_22_25 1, 1, 0, 0, 0, 47, -3, 550, 0

#define OCL_COMPARE_23_0 0, 0, 0, 1, 1, 23, 23, 0, 0
#define OCL_COMPARE_23_1 0, 0, 0, 1, 1, 24, 22, 23, 23
#define OCL_COMPARE_23_2 0, 0, 0, 1, 1, 25, 21, 46, 11
#define OCL_COMPARE_23_3 0, 0, 0, 1, 1, 26, 20, 69, 7
#define OCL_COMPARE_23_4 0, 0, 0, 1, 1, 27, 19, 92, 5
#define OCL_COMPARE_23_5 0, 0, 0, 1, 1, 28, 18, 115, 4
#define OCL_COMPARE_23_6 0, 0, 0, 1, 1, 29, 17, 138, 3
#define OCL_COMPARE_23_7 0, 0, 0, 1, 1, 30, 16, 161, 3
#define OCL_COMPARE_23_8 0, 0, 0, 1, 1, 31, 15, 184, 2
#define OCL_COMPARE_23_9 0, 0, 0, 1, 1, 32, 14, 207, 2
#define OCL_COMPARE_23_10 0, 0, 0, 1, 1, 33, 13, 230, 2
#define OCL_COMPARE_23_11 0, 0, 0, 1, 1, 34, 12, 253, 2
#define OCL_COMPARE_23_12 0, 0, 0, 1, 1, 35, 11, 276, 1
#define OCL_COMPARE_23_13 0, 0, 0, 1, 1, 36, 10, 299, 1
#define OCL_COMPARE_23_14 0, 0, 0, 1, 1, 37, 9, 322, 1
#define OCL_COMPARE_23_15 0, 0, 0, 1, 1, 38, 8, 345, 1
#define OCL_COMPARE_23_16 0, 0, 0, 1, 1, 39, 7, 368, 1
#define OCL_COMPARE_23_17 0, 0, 0, 1, 1, 40, 6, 391, 1
#define OCL_COMPARE_23_18 0, 0, 0, 1, 1, 41, 5, 414, 1
#define OCL_COMPARE_23_19 0, 0, 0, 1, 1, 42, 4, 437, 1
#define OCL_COMPARE_23_20 0, 0, 0, 1, 1, 43, 3, 460, 1
#define OCL_COMPARE_23_21 0, 0, 0, 1, 1, 44, 2, 483, 1
#define OCL_COMPARE_23_22 0, 0, 0, 1, 1, 45, 1, 506, 1
#define OCL_COMPARE_23_23 0, 1, 1, 1, 0, 46, 0, 529, 1
#define OCL_COMPARE_23_24 1, 1, 0, 0, 0, 47, -1, 552, 0
#define OCL_COMPARE_23_25 1, 1, 0, 0, 0, 48, -2, 575, 0

#define OCL_COMPARE_24_0 0, 0, 0, 1, 1, 24, 24, 0, 0
#define OCL_COMPARE_24_1 0, 0, 0, 1, 1, 25, 23, 24, 24
#define OCL_COMPARE_24_2 0, 0, 0, 1, 1, 26, 22, 48, 12
#define OCL_COMPARE_24_3 0, 0, 0, 1, 1, 27, 21, 72, 8
#define OCL_COMPARE_24_4 0, 0, 0, 1, 1, 28, 20, 96, 6
#define OCL_COMPARE_24_5 0, 0, 0, 1, 1, 29, 19, 120, 4
#define OCL_COMPARE_24_6 0, 0, 0, 1, 1, 30, 18, 144, 4
#define OCL_COMPARE_24_7 0, 0, 0, 1, 1, 31, 17, 168, 3
#define OCL_COMPARE_24_8 0, 0, 0, 1, 1, 32, 16, 192, 3
#define OCL_COMPARE_24_9 0, 0, 0, 1, 1, 33, 15, 216, 2
#define OCL_COMPARE_24_10 0, 0, 0, 1, 1, 34, 14, 240, 2
#define OCL_COMPARE_24_11 0, 0, 0, 1, 1, 35, 13, 264, 2
#define OCL_COMPARE_24_12 0, 0, 0, 1, 1, 36, 12, 288, 2
#define OCL_COMPARE_24_13 0, 0, 0, 1, 1, 37, 11, 312, 1
#define OCL_COMPARE_24_14 0, 0, 0, 1, 1, 38, 10, 336, 1
#define OCL_COMPARE_24_15 0, 0, 0, 1, 1, 39, 9, 360, 1
#define OCL_COMPARE_24_16 0, 0, 0, 1, 1, 40, 8, 384, 1
#define OCL_COMPARE_24_17 0, 0, 0, 1, 1, 41, 7, 408, 1
#define OCL_COMPARE_24_18 0, 0, 0, 1, 1, 42, 6, 432, 1
#define OCL_COMPARE_24_19 0, 0, 0, 1, 1, 43, 5, 456, 1
#define OCL_COMPARE_24_20 0, 0, 0, 1, 1, 44, 4, 480, 1
#define OCL_COMPARE_24_21 0, 0, 0, 1, 1, 45, 3, 504, 1
#define OCL_COMPARE_24_22 0, 0, 0, 1, 1, 46, 2, 528, 1
#define OCL_COMPARE_24_23 0, 0, 0, 1, 1, 47, 1, 552, 1
#define OCL_COMPARE_24_24 0, 1, 1, 1, 0, 48, 0, 576, 1
#define OCL_COMPARE_24_25 1, 1, 0, 0, 0, 49, -1, 600, 0

#define OCL_COMPARE_25_0 0, 0, 0, 1, 1, 25, 25, 0, 0
#define OCL_COMPARE_25_1 0, 0, 0, 1, 1, 26, 24, 25, 25
#define OCL_COMPARE_25_2 0, 0, 0, 1, 1, 27, 23, 50, 12
#define OCL_COMPARE_25_3 0, 0, 0, 1, 1, 28, 22, 75, 8
#define OCL_COMPARE_25_4 0, 0, 0, 1, 1, 29, 21, 100, 6
#define OCL_COMPARE_25_5 0, 0, 0, 1, 1, 30, 20, 125, 5
#define OCL_COMPARE_25_6 0, 0, 0, 1, 1, 31, 19, 150, 4
#define OCL_COMPARE_25_7 0, 0, 0, 1, 1, 32, 18, 175, 3
#define OCL_COMPARE_25_8 0, 0, 0, 1, 1, 33, 17, 200, 3
#define OCL_COMPARE_25_9 0, 0, 0, 1, 1, 34, 16, 225, 2
#define OCL_COMPARE_25_10 0, 0, 0, 1, 1, 35, 15, 250, 2
#define OCL_COMPARE_25_11 0, 0, 0, 1, 1, 36, 14, 275, 2
#define OCL_COMPARE_25_12 0, 0, 0, 1, 1, 37, 13, 300, 2
#define OCL_COMPARE_25_13 0, 0, 0, 1, 1, 38, 12, 325, 1
#define OCL_COMPARE_25_14 0, 0, 0, 1, 1, 39, 11, 350, 1
#define OCL_COMPARE_25_15 0, 0, 0, 1, 1, 40, 10, 375, 1
#define OCL_COMPARE_25_16 0, 0, 0, 1, 1, 41, 9, 400, 1
#define OCL_COMPARE_25_17 0, 0, 0, 1, 1, 42, 8, 425, 1
#define OCL_COMPARE_25_18 0, 0, 0, 1, 1, 43, 7, 450, 1
#define OCL_COMPARE_25_19 0, 0, 0, 1, 1, 44, 6, 475, 1
#define OCL_COMPARE_25_20 0, 0, 0, 1, 1, 45, 5, 500, 1
#define OCL_COMPARE_25_21 0, 0, 0, 1, 1, 46, 4, 525, 1
#define OCL_COMPARE_25_22 0, 0, 0, 1, 1, 47, 3, 550, 1
#define OCL_COMPARE_25_23 0, 0, 0, 1, 1, 48, 2, 575, 1
#define OCL_COMPARE_25_24 0, 0, 0, 1, 1, 49, 1, 600, 1
#define OCL_COMPARE_25_25 0, 1, 1, 1, 0, 50, 0, 625, 1

#define OCL_PARAM_1(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) ARG1
#define OCL_PARAM_2(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) ARG2
#define OCL_PARAM_3(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) ARG3
#define OCL_PARAM_4(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) ARG4
#define OCL_PARAM_5(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) ARG5
#define OCL_PARAM_6(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) ARG6
#define OCL_PARAM_7(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) ARG7
#define OCL_PARAM_8(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) ARG8
#define OCL_PARAM_9(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7, ARG8, ARG9) ARG9

#define OCL_PARAM(N, ...)                  OCL_PARAM_##N(__VA_ARGS__)
#define OCL_COMPARE_OPERATOR3(ARG, N1, N2) OCL_PARAM( ARG , OCL_COMPARE_##N1##_##N2 )
#define OCL_COMPARE_OPERATOR2(ARG, N1, N2) OCL_COMPARE_OPERATOR3(ARG, N1, N2)
#define OCL_COMPARE_OPERATOR(ARG, N1, N2)  OCL_COMPARE_OPERATOR2(ARG, N1, N2)

#define OCL_LT(N1,N2)  OCL_COMPARE_OPERATOR( 1 , N1 , N2 )
#define OCL_LTE(N1,N2) OCL_COMPARE_OPERATOR( 2 , N1 , N2 )
#define OCL_EQ(N1,N2)  OCL_COMPARE_OPERATOR( 3 , N1 , N2 )
#define OCL_NEQ(N1,N2) OCL_NOT( OCL_COMPARE_OPERATOR( 3 , N1 , N2 ) )
#define OCL_GTE(N1,N2) OCL_COMPARE_OPERATOR( 4 , N1 , N2 )
#define OCL_GT(N1,N2)  OCL_COMPARE_OPERATOR( 5 , N1 , N2 )

#define OCL_ADD(N1,N2)  OCL_COMPARE_OPERATOR( 6 , N1 , N2 )
#define OCL_SUB(N1,N2)  OCL_COMPARE_OPERATOR( 7 , N1 , N2 )
#define OCL_MULT(N1,N2) OCL_COMPARE_OPERATOR( 8 , N1 , N2 )
#define OCL_DIV(N1,N2)  OCL_COMPARE_OPERATOR( 9 , N1 , N2 )
//==============================================================================


//---[ OCL_FOR LOOP ]---------------------------------------------------------------
#define OCL_MAX_FOR_LOOPS 25

#define OCL_ARGS_DO_EXPR(EXPR_, ITER, ...) EXPR_(ITER, __VA_ARGS__ )
#define OCL_ARGS_ITER_1(OFF, EXPR_, ...)                                            OCL_ARGS_DO_EXPR( EXPR_ , OFF              , __VA_ARGS__ )
#define OCL_ARGS_ITER_2(OFF, EXPR_, ...)  OCL_ARGS_ITER_1(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 1 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_3(OFF, EXPR_, ...)  OCL_ARGS_ITER_2(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 2 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_4(OFF, EXPR_, ...)  OCL_ARGS_ITER_3(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 3 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_5(OFF, EXPR_, ...)  OCL_ARGS_ITER_4(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 4 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_6(OFF, EXPR_, ...)  OCL_ARGS_ITER_5(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 5 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_7(OFF, EXPR_, ...)  OCL_ARGS_ITER_6(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 6 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_8(OFF, EXPR_, ...)  OCL_ARGS_ITER_7(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 7 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_9(OFF, EXPR_, ...)  OCL_ARGS_ITER_8(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 8 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_10(OFF, EXPR_, ...) OCL_ARGS_ITER_9(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 9 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_11(OFF, EXPR_, ...) OCL_ARGS_ITER_10(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 10) , __VA_ARGS__ )
#define OCL_ARGS_ITER_12(OFF, EXPR_, ...) OCL_ARGS_ITER_11(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 11) , __VA_ARGS__ )
#define OCL_ARGS_ITER_13(OFF, EXPR_, ...) OCL_ARGS_ITER_12(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 12) , __VA_ARGS__ )
#define OCL_ARGS_ITER_14(OFF, EXPR_, ...) OCL_ARGS_ITER_13(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 13) , __VA_ARGS__ )
#define OCL_ARGS_ITER_15(OFF, EXPR_, ...) OCL_ARGS_ITER_14(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 14) , __VA_ARGS__ )
#define OCL_ARGS_ITER_16(OFF, EXPR_, ...) OCL_ARGS_ITER_15(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 15) , __VA_ARGS__ )
#define OCL_ARGS_ITER_17(OFF, EXPR_, ...) OCL_ARGS_ITER_16(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 16) , __VA_ARGS__ )
#define OCL_ARGS_ITER_18(OFF, EXPR_, ...) OCL_ARGS_ITER_17(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 17) , __VA_ARGS__ )
#define OCL_ARGS_ITER_19(OFF, EXPR_, ...) OCL_ARGS_ITER_18(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 18) , __VA_ARGS__ )
#define OCL_ARGS_ITER_20(OFF, EXPR_, ...) OCL_ARGS_ITER_19(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 19) , __VA_ARGS__ )
#define OCL_ARGS_ITER_21(OFF, EXPR_, ...) OCL_ARGS_ITER_20(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 20) , __VA_ARGS__ )
#define OCL_ARGS_ITER_22(OFF, EXPR_, ...) OCL_ARGS_ITER_21(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 21) , __VA_ARGS__ )
#define OCL_ARGS_ITER_23(OFF, EXPR_, ...) OCL_ARGS_ITER_22(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 22) , __VA_ARGS__ )
#define OCL_ARGS_ITER_24(OFF, EXPR_, ...) OCL_ARGS_ITER_23(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 23) , __VA_ARGS__ )
#define OCL_ARGS_ITER_25(OFF, EXPR_, ...) OCL_ARGS_ITER_24(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR( EXPR_ , OCL_ADD(OFF, 24) , __VA_ARGS__ )

#define OCL_ARGS_FOR4(OFF, END, EXPR_, ...)   OCL_ARGS_ITER_##END(OFF, EXPR_, __VA_ARGS__)
#define OCL_ARGS_FOR3(OFF, END, EXPR_, ...)   OCL_ARGS_FOR4(OFF, END, EXPR_, __VA_ARGS__)
#define OCL_ARGS_FOR2(START, END, EXPR_, ...) OCL_ARGS_FOR3(START, OCL_INC(OCL_SUB(END,START)), EXPR_, __VA_ARGS__)

#define OCL_ARGS_FOR_CHECK_1 OCL_ARGS_FOR2
#define OCL_ARGS_FOR_CHECK_0 OCL_VOID_MACRO

#define OCL_ARGS_FOR_CHECK2(BOOL, START, END, EXPR_, ...) OCL_ARGS_FOR_CHECK_##BOOL(START, END, EXPR_, __VA_ARGS__)
#define OCL_ARGS_FOR_CHECK(BOOL, START, END, EXPR_, ...)  OCL_ARGS_FOR_CHECK2(BOOL, START, END, EXPR_, __VA_ARGS__)

#define OCL_ARGS_FOR(START, END, EXPR_, ...) OCL_ARGS_FOR_CHECK( OCL_LTE(START, END) , START, END, EXPR_, __VA_ARGS__)

#define OCL_DO_EXPR(EXPR_, ITER) EXPR_(ITER)
#define OCL_ITER_1(OFF, EXPR_)                          OCL_DO_EXPR( EXPR_ , OFF )
#define OCL_ITER_2(OFF, EXPR_)  OCL_ITER_1(OFF, EXPR_)  OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 1 ) )
#define OCL_ITER_3(OFF, EXPR_)  OCL_ITER_2(OFF, EXPR_)  OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 2 ) )
#define OCL_ITER_4(OFF, EXPR_)  OCL_ITER_3(OFF, EXPR_)  OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 3 ) )
#define OCL_ITER_5(OFF, EXPR_)  OCL_ITER_4(OFF, EXPR_)  OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 4 ) )
#define OCL_ITER_6(OFF, EXPR_)  OCL_ITER_5(OFF, EXPR_)  OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 5 ) )
#define OCL_ITER_7(OFF, EXPR_)  OCL_ITER_6(OFF, EXPR_)  OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 6 ) )
#define OCL_ITER_8(OFF, EXPR_)  OCL_ITER_7(OFF, EXPR_)  OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 7 ) )
#define OCL_ITER_9(OFF, EXPR_)  OCL_ITER_8(OFF, EXPR_)  OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 8 ) )
#define OCL_ITER_10(OFF, EXPR_) OCL_ITER_9(OFF, EXPR_)  OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 9 ) )
#define OCL_ITER_11(OFF, EXPR_) OCL_ITER_10(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 10) )
#define OCL_ITER_12(OFF, EXPR_) OCL_ITER_11(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 11) )
#define OCL_ITER_13(OFF, EXPR_) OCL_ITER_12(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 12) )
#define OCL_ITER_14(OFF, EXPR_) OCL_ITER_13(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 13) )
#define OCL_ITER_15(OFF, EXPR_) OCL_ITER_14(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 14) )
#define OCL_ITER_16(OFF, EXPR_) OCL_ITER_15(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 15) )
#define OCL_ITER_17(OFF, EXPR_) OCL_ITER_16(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 16) )
#define OCL_ITER_18(OFF, EXPR_) OCL_ITER_17(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 17) )
#define OCL_ITER_19(OFF, EXPR_) OCL_ITER_18(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 18) )
#define OCL_ITER_20(OFF, EXPR_) OCL_ITER_19(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 19) )
#define OCL_ITER_21(OFF, EXPR_) OCL_ITER_20(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 20) )
#define OCL_ITER_22(OFF, EXPR_) OCL_ITER_21(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 21) )
#define OCL_ITER_23(OFF, EXPR_) OCL_ITER_22(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 22) )
#define OCL_ITER_24(OFF, EXPR_) OCL_ITER_23(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 23) )
#define OCL_ITER_25(OFF, EXPR_) OCL_ITER_24(OFF, EXPR_) OCL_DO_EXPR( EXPR_ , OCL_ADD(OFF, 24) )

#define OCL_FOR4(OFF, END, EXPR_)   OCL_ITER_##END(OFF, EXPR_)
#define OCL_FOR3(OFF, END, EXPR_)   OCL_FOR4(OFF, END, EXPR_)
#define OCL_FOR2(START, END, EXPR_) OCL_FOR3(START, OCL_INC(OCL_SUB(END,START)), EXPR_)

#define OCL_FOR_CHECK_1 OCL_FOR2
#define OCL_FOR_CHECK_0 OCL_VOID_MACRO

#define OCL_FOR_CHECK2(BOOL, START, END, EXPR_) OCL_FOR_CHECK_##BOOL(START, END, EXPR_)
#define OCL_FOR_CHECK(BOOL, START, END, EXPR_)  OCL_FOR_CHECK2(BOOL, START, END, EXPR_)

#define OCL_FOR(START, END, EXPR_) OCL_FOR_CHECK( OCL_LTE(START, END) , START, END, EXPR_)
//==============================================================================


//---[ OCL_FOR LOOP 2 ]-------------------------------------------------------------
#define OCL_ARGS_DO_EXPR_2(EXPR_, ITER, ...) EXPR_(ITER, __VA_ARGS__ )
#define OCL_ARGS_ITER_2_1(OFF, EXPR_, ...)                                              OCL_ARGS_DO_EXPR_2( EXPR_ , OFF          , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_2(OFF, EXPR_, ...)  OCL_ARGS_ITER_2_1(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 1 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_3(OFF, EXPR_, ...)  OCL_ARGS_ITER_2_2(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 2 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_4(OFF, EXPR_, ...)  OCL_ARGS_ITER_2_3(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 3 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_5(OFF, EXPR_, ...)  OCL_ARGS_ITER_2_4(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 4 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_6(OFF, EXPR_, ...)  OCL_ARGS_ITER_2_5(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 5 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_7(OFF, EXPR_, ...)  OCL_ARGS_ITER_2_6(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 6 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_8(OFF, EXPR_, ...)  OCL_ARGS_ITER_2_7(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 7 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_9(OFF, EXPR_, ...)  OCL_ARGS_ITER_2_8(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 8 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_10(OFF, EXPR_, ...) OCL_ARGS_ITER_2_9(OFF, EXPR_, __VA_ARGS__)  OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 9 ) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_11(OFF, EXPR_, ...) OCL_ARGS_ITER_2_10(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 10) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_12(OFF, EXPR_, ...) OCL_ARGS_ITER_2_11(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 11) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_13(OFF, EXPR_, ...) OCL_ARGS_ITER_2_12(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 12) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_14(OFF, EXPR_, ...) OCL_ARGS_ITER_2_13(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 13) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_15(OFF, EXPR_, ...) OCL_ARGS_ITER_2_14(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 14) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_16(OFF, EXPR_, ...) OCL_ARGS_ITER_2_15(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 15) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_17(OFF, EXPR_, ...) OCL_ARGS_ITER_2_16(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 16) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_18(OFF, EXPR_, ...) OCL_ARGS_ITER_2_17(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 17) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_19(OFF, EXPR_, ...) OCL_ARGS_ITER_2_18(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 18) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_20(OFF, EXPR_, ...) OCL_ARGS_ITER_2_19(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 19) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_21(OFF, EXPR_, ...) OCL_ARGS_ITER_2_20(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 20) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_22(OFF, EXPR_, ...) OCL_ARGS_ITER_2_21(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 21) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_23(OFF, EXPR_, ...) OCL_ARGS_ITER_2_22(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 22) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_24(OFF, EXPR_, ...) OCL_ARGS_ITER_2_23(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 23) , __VA_ARGS__ )
#define OCL_ARGS_ITER_2_25(OFF, EXPR_, ...) OCL_ARGS_ITER_2_24(OFF, EXPR_, __VA_ARGS__) OCL_ARGS_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 24) , __VA_ARGS__ )

#define OCL_ARGS_FOR_2_4(OFF, END, EXPR_, ...)   OCL_ARGS_ITER_2_##END(OFF, EXPR_, __VA_ARGS__)
#define OCL_ARGS_FOR_2_3(OFF, END, EXPR_, ...)   OCL_ARGS_FOR_2_4(OFF, END, EXPR_, __VA_ARGS__)
#define OCL_ARGS_FOR_2_2(START, END, EXPR_, ...) OCL_ARGS_FOR_2_3(START, OCL_INC(OCL_SUB(END,START)), EXPR_, __VA_ARGS__)

#define OCL_ARGS_FOR_CHECK_2_1 OCL_ARGS_FOR_2_2
#define OCL_ARGS_FOR_CHECK_2_0 OCL_VOID_MACRO

#define OCL_ARGS_FOR_CHECK2_2(BOOL, START, END, EXPR_, ...) OCL_ARGS_FOR_CHECK_2_##BOOL(START, END, EXPR_, __VA_ARGS__)
#define OCL_ARGS_FOR_CHECK_2(BOOL, START, END, EXPR_, ...)  OCL_ARGS_FOR_CHECK2_2(BOOL, START, END, EXPR_, __VA_ARGS__)

#define OCL_ARGS_FOR_2(START, END, EXPR_, ...) OCL_ARGS_FOR_CHECK_2( OCL_LTE(START, END) , START, END, EXPR_, __VA_ARGS__)

#define OCL_DO_EXPR_2(EXPR_, ITER) EXPR_(ITER)
#define OCL_ITER_2_1(OFF, EXPR_)                            OCL_DO_EXPR_2( EXPR_ , OFF )
#define OCL_ITER_2_2(OFF, EXPR_)  OCL_ITER_2_1(OFF, EXPR_)  OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 1 ) )
#define OCL_ITER_2_3(OFF, EXPR_)  OCL_ITER_2_2(OFF, EXPR_)  OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 2 ) )
#define OCL_ITER_2_4(OFF, EXPR_)  OCL_ITER_2_3(OFF, EXPR_)  OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 3 ) )
#define OCL_ITER_2_5(OFF, EXPR_)  OCL_ITER_2_4(OFF, EXPR_)  OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 4 ) )
#define OCL_ITER_2_6(OFF, EXPR_)  OCL_ITER_2_5(OFF, EXPR_)  OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 5 ) )
#define OCL_ITER_2_7(OFF, EXPR_)  OCL_ITER_2_6(OFF, EXPR_)  OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 6 ) )
#define OCL_ITER_2_8(OFF, EXPR_)  OCL_ITER_2_7(OFF, EXPR_)  OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 7 ) )
#define OCL_ITER_2_9(OFF, EXPR_)  OCL_ITER_2_8(OFF, EXPR_)  OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 8 ) )
#define OCL_ITER_2_10(OFF, EXPR_) OCL_ITER_2_9(OFF, EXPR_)  OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 9 ) )
#define OCL_ITER_2_11(OFF, EXPR_) OCL_ITER_2_10(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 10) )
#define OCL_ITER_2_12(OFF, EXPR_) OCL_ITER_2_11(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 11) )
#define OCL_ITER_2_13(OFF, EXPR_) OCL_ITER_2_12(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 12) )
#define OCL_ITER_2_14(OFF, EXPR_) OCL_ITER_2_13(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 13) )
#define OCL_ITER_2_15(OFF, EXPR_) OCL_ITER_2_14(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 14) )
#define OCL_ITER_2_16(OFF, EXPR_) OCL_ITER_2_15(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 15) )
#define OCL_ITER_2_17(OFF, EXPR_) OCL_ITER_2_16(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 16) )
#define OCL_ITER_2_18(OFF, EXPR_) OCL_ITER_2_17(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 17) )
#define OCL_ITER_2_19(OFF, EXPR_) OCL_ITER_2_18(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 18) )
#define OCL_ITER_2_20(OFF, EXPR_) OCL_ITER_2_19(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 19) )
#define OCL_ITER_2_21(OFF, EXPR_) OCL_ITER_2_20(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 20) )
#define OCL_ITER_2_22(OFF, EXPR_) OCL_ITER_2_21(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 21) )
#define OCL_ITER_2_23(OFF, EXPR_) OCL_ITER_2_22(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 22) )
#define OCL_ITER_2_24(OFF, EXPR_) OCL_ITER_2_23(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 23) )
#define OCL_ITER_2_25(OFF, EXPR_) OCL_ITER_2_24(OFF, EXPR_) OCL_DO_EXPR_2( EXPR_ , OCL_ADD(OFF, 24) )

#define OCL_FOR_2_4(OFF, END, EXPR_) OCL_ITER_2_##END(OFF, EXPR_)
#define OCL_FOR_2_3(OFF, END, EXPR_) OCL_FOR_2_4(OFF, END, EXPR_)
#define OCL_FOR_2_2(START, END, EXPR_) OCL_FOR_2_3(START, OCL_INC(OCL_SUB(END,START)), EXPR_)

#define OCL_FOR_CHECK_2_1 OCL_FOR_2_2
#define OCL_FOR_CHECK_2_0 OCL_VOID_MACRO

#define OCL_FOR_CHECK_2_2(BOOL, START, END, EXPR_) OCL_FOR_CHECK_2_##BOOL(START, END, EXPR_)
#define OCL_FOR_CHECK_2(BOOL, START, END, EXPR_)  OCL_FOR_CHECK_2_2(BOOL, START, END, EXPR_)

#define OCL_FOR_2(START, END, EXPR_) OCL_FOR_CHECK_2( OCL_LTE(START, END) , START, END, EXPR_)
//==============================================================================


//---[ USEFUL STUFF ]-----------------------------------------------------------
#define OCL_PRINT_WIDTH 30
#define OCL_PRINT(X) std::cout << std::left << std::setw(PRINT_WIDTH) << #X " : " << X << std::endl
//==============================================================================

#endif
