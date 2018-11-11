# Code to convert back and forth between fixed point numbers, float point numbers and posits

import torch
import math

num_bits_fixed_point = 8
num_frac_bits = 5

def reverse_string(input_string):
    new_string = ""
    for i in range(len(input_string)):
        index = len(input_string) -1 - i
        val = input_string[index]
        new_string += str(val)
    return new_string

def convert_int_to_bin(float_num):
    bin_div = 2
    
    bit_string = []
    result = float_num
    remainder = 0
    
    result = int(result)
    bin_div = int(bin_div) #TODO: might be unnecessary
    #while(result >= useed):
    while(result > 0):
        #print("result",result, "useed", useed)
        result_div = int(result/bin_div)
        #print("result", result, "result_div", result_div, "result_div * useed", result_div * useed)
        remainder = result - (result_div * bin_div )
        #print("appending remainder: ", remainder)
        result = result_div
        bit_string.append(remainder)
    new_string = reverse_string(bit_string)
    return new_string

def compute_useed_power(float_num, useed,es):

    bit_string = []
    result = float_num

    remainder = 0
    
    result = int(result)
    useed = int(useed)
    while(result > useed):
        result_div = result/useed
        result_div = int(result_div)
        #print("result_div", result_div)
        remainder = result - (result_div * useed )
        #print("remainder", remainder)
        result = result_div
        if(remainder > 1):
            remainder = convert_int_to_bin(remainder)
        bit_string.append(remainder)
    new_string = reverse_string(bit_string)

    return new_string[0:es], str(remainder)

def convert_fractional_decimal_to_bin(fractional_decimal, num_frac_bits):
    result = fractional_decimal
    decimals = ""
    bits_available = num_frac_bits
    
    while bits_available > 0:

        result = result * 2
        if(result < 1):
            decimals+= "0"
        else:
            decimals+= "1"
            result = result - 1
        
        bits_available -= 1
    
    #new_string =  decimals[-1::-1]
    return decimals

"""
def convert_fractional_part(fp_value, useed):
    result = fp_value
    decimals = ""
    while (result < 1):
        result = result * 2
        decimals+= "0"
    
    #TODO: convert this number of decimals from constant(3) to flexible
    decimals+= "1"
    fractional = result - 1
    for i in range(3):
        new_res = fractional * 2
        if new_res < 1 : break
        decimals+= "1"
        fractional = new_res - 1

    new_string =  decimals[-1::-1]
    return new_string,new_string
"""

def get_useed_exponent(num_regime_bits, regime_sign):
    regime_bits = ""
    if(regime_sign == -1):
        for i in range(num_regime_bits):
            regime_bits += "0"
        regime_bits += "1"
    else:
        for i in range(num_regime_bits):
            regime_bits += "1"
        regime_bits += "0"
    #print("regime_bits", regime_bits)
    return regime_bits

def get_new_exponent(integral):
    powered = 2 ** 0
    counter = 0
    while powered < integral :
        #print("powered", powered, "integral", integral, "counter", counter )
        counter += 1
        powered = 2 ** counter
    return counter-1

#Reference: posit paper section 4 "Converting values into posits"
def convert_float_to_posit(fp_value, nbits, es):
    #nbits = wordsize #es - maximum size of exponent field
    
    exponent_bits, regime_bits, fraction_bits = (0,0,0)
    #TODO: Determine the above parameters dynamically from the value of the float num provided

    sign_bit = "0"
    if (fp_value <= 0):
        sign_bit = "1"
    
    if (fp_value == 0):
        return "0000000000000000"
        #TODO: dynamically create string of size nbits
    if (fp_value == math.inf):
        return "1000000000000000"

    posit_string = ""
    useed = 2**(2**es)
    
    #print("useed", useed)
    
    #useed_exponent = get_useed_exponent(regime_bits, regime_sign)
    #useed_exponent = get_useed_exponent(3, 1)
    useed_power = ""

    if (fp_value < 1):
        useed_power =  get_useed_exponent(3, -1)
    else:
        #useed_power, fraction = compute_useed_power(fp_value, useed,es)
        useed_power =  get_useed_exponent(3, 1)

    integral = fp_value/(useed **(-3))
    print("integral", integral)
    new_exponent = get_new_exponent(integral)

    #print("new_exponent: ", new_exponent)
    fraction_bits = ""

    #TODO: The 2 param is redundant, change function signature
    new_exponent_bits = convert_int_to_bin(new_exponent)
    
    integral_remainder = integral - (2**new_exponent)
    
    fractional_integer = (useed/(2** new_exponent)) * integral_remainder
    print("fractional_integer", fractional_integer)
    fraction_bits = convert_int_to_bin(fractional_integer)
    #print("fraction_bits", fraction_bits)
        
    #print("sign_bit: ", sign_bit, "useed_power: ",useed_power, "exponent: ", new_exponent_bits, "fraction_bits: ",fraction_bits)
        
    posit_string = sign_bit+useed_power+new_exponent_bits+fraction_bits
    return posit_string

def convert_bin_to_int(bin_string):
    len_bin = len(bin_string)
    sum = 0
    for i in range(len_bin):
        power = len_bin - 1 - i
        ind_val = int(bin_string[i])
        val = ind_val * (2 ** power)
        sum += val
    return sum

def all_zero(posit_str):
    for i in range(len(posit_str)):
        if int(posit_str[i]) == 1: return False
    return True


#Reference: posit paper section 2
def convert_posit_to_float(es, posit_string):
    
    if all_zero(posit_string):
        print("expression_val", 0)
        return 0
    
    if int(posit_string[0]) == 1 and all_zero(posit_string[1:]):
        print("expression_val", math.inf)
        return 0

    #TODO: Extract regime bits dynamically
    sign = -1 if int(posit_string[0]) == 1 else 1
    scale_power_or_useed = 2**(2**es)
    scale_sign =  -1 if int(posit_string[es+1]) == 1 else 1
    first_scale_factor = scale_power_or_useed ** (scale_sign * es)
    
    end_exponent_bits_index = es+2+es
    
    exponent_bits = posit_string[int(es+2):int(end_exponent_bits_index)]
    exponent_val = convert_bin_to_int(exponent_bits)
    second_scale_factor = 2**exponent_val
    fraction_bits = posit_string[end_exponent_bits_index:len(posit_string)]
    fraction_val = convert_bin_to_int(fraction_bits)
    fraction_factor = (1 + float(fraction_val)/scale_power_or_useed)
    
    #print("first_scale_factor: ", first_scale_factor, "second_scale_factor: ", second_scale_factor, "last_factor: ", fraction_factor  )
    
    expression_val = sign * first_scale_factor * second_scale_factor * fraction_factor
    #print("expression_val", expression_val)

    return expression_val

def convert_fixed_point_to_float(fixed_point_num):

    #num_bits_fixed_point
    #print("fixed_point_num: ", fixed_point_num)
    
    fixed_point_num_string = str(fixed_point_num)
    sign = int(fixed_point_num_string[0])
    #print("sign: ", sign)
    fixed_point_num = fixed_point_num_string[1:]
    #print("fixed_point_num: ", fixed_point_num)
    radix_index = fixed_point_num.index('.')
    #print("radix_index: ", radix_index)
    integral_part = fixed_point_num[0:radix_index]
    #print("integral_part: ", integral_part)
    fractional_part = fixed_point_num[radix_index+1:]
    #print("fractional_part: ", fractional_part)
    
    integral_part_reversed =  reverse_string(integral_part)
    
    integral_sum = 0
    for i in range(len(integral_part_reversed)):
        num_read = int(integral_part_reversed[i])
        #print("num_read: ", num_read)
        if num_read == 1:
            #print("adding: ", 2**i )
            integral_sum += 2**i
    #print("integral_sum: ", integral_sum)

    fractional_sum = 0
    for j in range(len(fractional_part)):
        num_read = int(fractional_part[j])
        #print("num_read: ", num_read)
        if num_read == 1:
            #print("adding: ", float(1)/2**(j+1) )
            fractional_sum += float(1)/2**(j+1)
    #print("fractional_sum: ", fractional_sum)

    total_float = integral_sum + fractional_sum

    if sign == 1:
        total_float = total_float * -1

    return total_float
    
def convert_float_to_fixed_point(float_num):
    fp_string = ""
    abso_val = float_num
    if float_num < 0:
        fp_string += "1"
        abso_val = abso_val * -1
    else:
        fp_string += "0"

    fractional_part = 0
    integral_part = 0
    if abso_val < 1:
        fractional_part = abso_val
    else:
        integral_part = int(abso_val)
        fractional_part = abso_val - integral_part

    int_bin_string = convert_int_to_bin(integral_part)
    frac_bin_string = convert_fractional_decimal_to_bin(fractional_part, num_frac_bits)

    return fp_string + int_bin_string +"." + frac_bin_string


def test_conversions2():
    
    test_numbers = [3.553926946732666e-06]
    num = 3.553926946732666e-06
    for i in range(15):
        posit_value = convert_float_to_posit( num, 16, i)
        print("onbtained posit value: ", posit_value,"\n" )


def test_conversions():

    #test_numbers = [3.553926944732666e-06, 3.553926946732666e-06, 3.553926946732666e-04]
    test_numbers = [3.553926944732666e-06, 3.553926946732666e-06]

    for i in range(len(test_numbers)):
        print("\n-------------------------")
        num = test_numbers[i]
        print("converting float ", num, " to posit" )
        posit_value = convert_float_to_posit( num, 16, 3)
        print("onbtained posit value: ", posit_value )
        print("converting posit ",posit_value, "back to float " )
        float_point_value = convert_posit_to_float(3, posit_value)
        print("onbtained float point value: ", float_point_value )
        print("diff: ", num - float_point_value )
        print("-------------------------\n")

#test_conversions2()
#print(reverse_string("110111111001110"))

#print(convert_fixed_point_to_float("11001.0111"))
print(convert_float_to_fixed_point(-9.124343757546456))

print(convert_fixed_point_to_float("11001.00011"))


#print(convert_posit_to_float(3, "0000110111011101"))
#print(convert_float_to_posit( 3.553926944732666e-06, 16, 3))


#TODO: Integrate with quantization after
def quantize_(x, qi, qf):
    fmax = 1. - float(torch.pow(2., torch.FloatTensor([-1. * qf])).numpy()[0])
    imax =      float((torch.pow(2., torch.FloatTensor([qi-1])) - 1).numpy()[0])
    imin = -1 * float((torch.pow(2., torch.FloatTensor([qi-1]))).numpy()[0])
    fdiv = float(torch.pow(2., torch.FloatTensor([-qf])).numpy()[0])
    
    x = torch.floor ( x / fdiv) * fdiv
    x = torch.clamp(x, imin, imax + fmax)
    
    return x

def quantize_sparse_weights(x, qi, qf):
    pos_weights_mask   = torch.gt(x, 0)
    pos_weights_values = torch.masked_select(x, pos_weights_mask)
    
    pos_min = pos_weights_values.min()
    pos_max = pos_weights_values.max()
    
    pos_weights = x * pos_weights_mask.float()
    pos_weights = pos_weights - pos_min
    new_max = pos_weights.max()
    pos_weights = torch.div ( pos_weights, new_max )
    pos_weights = pos_weights * pos_weights_mask.float()
    
    pos_weights = quantize_(pos_weights, qi, qf)
    pos_weights = torch.mul(pos_weights , new_max)
    pos_weights = pos_weights + pos_min
    pos_weights = pos_weights * pos_weights_mask.float()
    
    return pos_weights

num_full = torch.FloatTensor ([3.78643467])
g_int = 2
g_float = 3

#print(quantize_(num_full, g_int, g_float))
