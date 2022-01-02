# lstsq_eigs.py
"""Least Squares and Computing Eigenvalues.
Sam Goldrup
30 October 2021
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath

def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q, R = la.qr(A, mode="economic")
    Q_T_b = np.matmul(Q.T, b)

    n=len(Q_T_b)
    x=np.array([0.0 for i in range(n)]) #initialize an array of 0's
    for i in range(n-1,-1,-1): #iterate backwards from n-1th column to 0th column
        r=(Q_T_b[i]-sum([x[j]*R[i][j] for j in range(i+1,n)]))/R[i][i]
        x[i]=r #assign elements of the soln vector
    return x

def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    housing_data = np.load("housing.npy") #load in the data
    len_data = len(housing_data) #number of rows
    b = housing_data[:,1] #prices
    A = np.column_stack((housing_data[:,0], np.ones(len_data))) #build A matrix
    solns = least_squares(A, b) #get slope and intercept
    slope, intercept = solns
    plt.plot(housing_data[:,0], b, 'k.', label="homes") #plot data pts, then plot line
    plt.plot(housing_data[:,0], intercept + slope*housing_data[:,0], 'b-', lw=2, label="fit")
    plt.xlabel("years since 2000")
    plt.ylabel("prices ($1000's)")
    plt.title("housing prices over time")
    plt.legend(loc="upper left")
    plt.show()


def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    housing_data = np.load("housing.npy")
    yrs = housing_data[:,0] #years column
    prices = housing_data[:,1] #prices column
    orders = (3,6,9,12) #orders of the polynomial
    num_points = 100 #num points to plot for the line

    A_mats = [np.vander(yrs, k+1) for k in orders] #make vandermonde matrices
    coeffs = [la.lstsq(A_mats[i], prices)[0] for i in range(len(A_mats))] #get sets of x vectors for each order
    print(A_mats)
    print(coeffs)
    
    domain = np.linspace(min(yrs),max(yrs),num_points) #the chosen domain, a bit hardcoding
    for i in range(len(orders)):
        plt.subplot(2,2,i+1) #2x2 grid of subplots
        plt.plot(yrs, prices, 'k.', label="homes") #plot data points
        y_vals = np.polyval(coeffs[i], domain) #projected y_vals to plot
        title = "fit of " + str(orders[i]) + " degree polynomial"
        plt.plot(domain, y_vals, 'b-', lw=2)
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200) #angles to plot from
    cos_t, sin_t = np.cos(theta), np.sin(theta) #get sines, cosines
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A) #get radii

    plt.plot(r*cos_t, r*sin_t)
    plt.title("fitting elliptical motion")
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.gca().set_aspect("equal", "datalim")

def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    xk, yk = np.load("ellipse.npy").T
    plt.plot(xk, yk, 'k*')
    x_sq = xk*xk #x^2
    x = xk
    xy = xk * yk
    y_sq = yk * yk #y^2
    y = yk
    b = np.ones(len(xk))
    
    A = np.column_stack((x_sq, x, xy, y, y_sq)) #build A matrix
    a, b, c, d, e = la.lstsq(A,b)[0] #get coeffs

    plot_ellipse(a,b,c,d,e) #plot it bb!
    plt.show()


def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m,n = np.shape(A) #A is a square so m=n bb!!
    x_0 = np.random.random(n) #random vector of length n
    x_0 /= la.norm(x_0) #normalize it
    x_curr = x_0 #set x_curr to the vector x_0
    for k in range(N): #iterate N times
        x_next = np.matmul(A, x_curr) #left mult x_curr by A
        x_next /= la.norm(x_next) #normalize it
        if la.norm(x_next - x_curr) < tol: 
            break #break if within tolerance
        else:
            x_curr = x_next #else, iterate again with x_curr updated
        print(x_next)
    x_fin = x_next
    return x_fin.T @ A @ x_fin, x_fin


def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m,n = np.shape(A)
    S = la.hessenberg(A) #put A in upp Hess form
    for k in range(N): #hessenberg preconditioning
        Q,R = la.qr(S) #the QR decomp of A
        S = R @ Q
    eigs = []
    i = 0
    while i < n:
        if i == n-1:
            eigs.append(S[i][i])
        elif abs(S[i+1][i]) < tol:
            eigs.append(S[i][i])
        elif abs(S[i+1][i]) > tol or np.allclose(abs(S[i+1][i]), tol) == True:
            T = S[i][i] + S[i+1][i+1] #get trace
            D = (S[i][i] * S[i+1][i+1]) - (S[i][i+1] * S[i+1][i]) #get det
            sqrt_terms = cmath.sqrt(T**2 - 4*D)
            first_eig = T/2 + sqrt_terms/2 #get eig 1
            second_eig = T/2 - sqrt_terms/2 #get the conjugate
            eigs.append(first_eig)
            eigs.append(second_eig)
            i += 1
        i += 1
    return eigs
