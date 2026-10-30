#ifndef SYMNMF_H
#define SYMNMF_H

double **allocate_matrix(int rows, int cols);
void free_matrix(double **mat, int rows);

void sym(double **X, double **A, int n, int d);
void ddg(double **A, double *D_diag, int n);
void norm(double **A, double *D_diag, double **W, int n);

void multiply_matrices(double **A, double **B, double **C, int m, int p, int n);
void multiply_matrices_transpose(double **A, double **B, double **C, int n, int k, int m);
double frobenius_norm_squared_diff(double **A, double **B, int n, int k);
void update_H(double **W, double **H, int n, int k, int max_iter, double epsilon);

#endif