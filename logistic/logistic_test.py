def Gradient_Ascent_test():
    def f_frime(x_old):
        return -2 * x_old + 4

    x_old = -1
    x_new = 10
    alpha = 0.1
    precision = 0.000001

    while abs(x_new - x_old) > precision:
        x_old = x_new
        x_new = x_old + alpha * f_frime(x_old)
        print(x_new)


if __name__ == '__main__':
    Gradient_Ascent_test()
