def costFunc(value, expectedValue):
    return (value - expectedValue)**2
    
def costPrime(value, expectedValue):
    return 2 * (value - expectedValue)

costDic = {
    'squared': (costFunc, costPrime)
}