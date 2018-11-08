# Code to convert back and forth between fixed point numbers and posits

import torch
import math

def convert_int_to_bin(fixed_num, useed):
    
    #print("\n-----------------")
    #print("fixed_num",fixed_num, "useed", useed)
    
    bit_string = []
    result = fixed_num
    remainder = 0
    
    result = int(result)
    useed = int(useed)
    #while(result >= useed):
    while(result > 0):
        #print("result",result, "useed", useed)
        result_div = int(result/useed)
        #print("result", result, "result_div", result_div, "result_div * useed", result_div * useed)
        remainder = result - (result_div * useed )
        #print("appending remainder: ", remainder)
        result = result_div
        bit_string.append(remainder)
  
    
    #TODO: improve/simplify string reversal
    #new_string = bit_string[-1::-1]
    #"""
    new_string = ""
    for i in range(len(bit_string)):
        index = len(bit_string) -1 - i
        val = bit_string[index]
        new_string += str(val)
    #"""
    #print("-----------------\n")
    return new_string, str(remainder)

def compute_useed_power(fixed_num, useed,es):

    bit_string = []
    result = fixed_num

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
            remainder, frc = convert_int_to_bin(remainder, 2)
        bit_string.append(remainder)
    #bit_string.append(remainder)

    #TODO: improve/simplify string reversal
    #new_string = bit_string[-1::-1]
    #"""
    for i in range(len(bit_string)):
        index = len(bit_string) -1 - i
        val = bit_string[index]
        new_string += str(val)
    #"""
    return new_string[0:es], str(remainder)

def convert_fractional_part(fp_value, useed):
    result = fp_value
    decimals = ""
    while (result < 1):
        result = result * 2
        decimals+= "0"
    
    #TODO: convert this number of decimals from fixed(3) to flexible
    decimals+= "1"
    fractional = result - 1
    for i in range(3):
        new_res = fractional * 2
        if new_res < 1 : break
        decimals+= "1"
        fractional = new_res - 1

    new_string =  decimals[-1::-1]
    return new_string,new_string

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
def convert_fixed_to_posit(fp_value, nbits, es):
    #nbits = wordsize #es - maximum size of exponent field
    
    exponent_bits, regime_bits, fraction_bits = (0,0,0)
    #TODO: Determine the above parameters dynamically from the value of the fixed num provided

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
    #print("integral", integral)
    new_exponent = get_new_exponent(integral)

    #print("new_exponent: ", new_exponent)
    fraction_bits = ""

    #TODO: The 2 param is redundant, change function signature
    new_exponent_bits, useless_frac = convert_int_to_bin(new_exponent,2)
    
    integral_remainder = integral - (2**new_exponent)
    
    fractional_integer = (useed/(2** new_exponent)) * integral_remainder
    #print("fractional_integer", fractional_integer)
    fraction_bits, uselessfrac = convert_int_to_bin(fractional_integer,2)
    #print("fraction_bits", fraction_bits)

    #other_power, fraction = convert_fractional_part(fp_value, useed)

    #exponent, bit_fraction = convert_int_to_bin(fp_value, 2)
    #fraction_bits, fracre = convert_int_to_bin(fraction, 2)
    
        
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
def convert_posit_to_fixed(es, posit_string):
    
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


test_numbers = [3.553926944732666e-06, 3.553926946732666e-06, 3.553926946732666e-04]

for i in range(len(test_numbers)):
    print("\n-------------------------")
    num = test_numbers[i]
    print("converting ", num, " to posit" )
    posit_value = convert_fixed_to_posit( num, 16, 3)
    print("onbtained posit value: ", posit_value )
    print("converting posit ",posit_value, "back to fixed " )
    fixed_point_value = convert_posit_to_fixed(3, posit_value)
    print("onbtained fixed point value: ", fixed_point_value )
    print("diff: ", num - fixed_point_value )
    print("-------------------------\n")



#print(convert_posit_to_fixed(3, "0000110111011101"))
#print(convert_fixed_to_posit( 3.553926944732666e-06, 16, 3))


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










