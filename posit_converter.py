# Code to convert back and forth between fixed point numbers, float point numbers and posits

import torch
import math

num_bits_fixed_point = 8
num_frac_bits = 5
config_num_int_bits = 2
posit_num_bits = 16

def reverse_string(input_string):
    new_string = ""
    for i in range(len(input_string)):
        index = len(input_string) -1 - i
        val = input_string[index]
        new_string += str(val)
    return new_string

def convert_int_to_bin(float_num, num_int_bits):
    bin_div = 2
    
    print("number being converted from floatnum to bin bits: ", float_num )
    
    #bit_string = []
    bit_string =""
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
        #bit_string.append(remainder)
        bit_string += str(remainder)
    
    print("bit_string before reversal: ", bit_string)
    if(len(bit_string) > num_int_bits):
        bit_string = bit_string[0:num_int_bits] #TODO: confirm if right substring
    new_string = reverse_string(bit_string)
    print("bit_string after reversal: ", new_string)
    return new_string

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

def convert_posit_to_fixed_point_num(posit_num, num_int_bits, num_frac_bits):
    float_val = convert_posit_to_float(3, posit_num)
    fixed_point_num = convert_float_to_fixed_point (float_val)
    return float_val

def convert_fixed_point_num_to_posit(fixed_point_num):
    float_val = convert_fixed_point_to_float(fixed_point_num)
    posit_num = convert_float_to_posit(float_val)
    return posit_num

def create_except_case_string(start_bit, num_bits):
    ret_str = start_bit
    for i in range(num_bits -1):
        ret_str += "0"
    return ret_str

def get_regime_bits(abso_val, useed ):

    regime_bit_string = ""
    if (abso_val > useed):
        result = abso_val
        while result > useed:
            result = result/useed

def calculate_fraction(remainders_list, useed):
    sum_frac = 0
    for i in range(len(remainders_list)):
        index = len(remainders_list) - i -1
        num = remainders_list[index]
        frac = float(num)/ (useed ** i)
        sum_frac += frac
    return sum_frac

def get_number_to_bin_exponent(dividend):
    bin_div, exp = (2, 0)
    result = int(dividend)
 
    while(result > 1):
        result = int(result/bin_div)
        exp += 1
    return exp

def get_exponent_and_frac_bits(abso_val,useed,es,nbits, regime_length):

    result = abso_val
    
    expo = get_number_to_bin_exponent(result)
    print("expo", expo)
    expo_bits = convert_int_to_bin(expo,es)
    print("expo_bits", expo_bits)
    
    mixed_fraction = float(result)/(2**expo)
    print("mixed_fraction", mixed_fraction)
    frac_remainder = mixed_fraction - 1
    print("frac_remainder", frac_remainder)
    useed_divider = frac_remainder * (useed)
    print("useed_divider", useed_divider)
    
    available_bits_for_frac_remainder = nbits - (len(str(expo_bits)) + regime_length + 2)
    print("available_bits_for_frac_remainder: ", available_bits_for_frac_remainder)
    fraction_bits = convert_int_to_bin(useed_divider, available_bits_for_frac_remainder) #TODO: check what happens to fractional part of frac
    #exponent_bits = convert_int_to_bin(result,es)
    
    return expo_bits, fraction_bits


#TODO: Reuse this function in big and small
def get_string_bits_regular(abso_val,useed,es,nbits):
    regime_length = 0
    exponent_bits, fraction_bits = get_exponent_and_frac_bits(abso_val,useed,es,nbits, regime_length)
    return regime_length, exponent_bits, fraction_bits

def get_string_bits_big(abso_val, useed,es,nbits):
    
    #print("\n getting big string")
    
    result = abso_val
    #print("result", result)
    
    regime_length = 0
    while result > useed:
        result_div = int(result/useed)
        #print("result_div", result_div)
        
        result = result_div
        regime_length += 1

    #print("regime_length: ", regime_length)
    dividend = float(abso_val)/(useed**regime_length)
    print("dividend: ", dividend)
    exponent_bits, fraction_bits = get_exponent_and_frac_bits(dividend,useed,es,nbits, regime_length)

    print("nbits: ", nbits, "regime_length",regime_length, "len exponent_bits", len(exponent_bits), "len fraction_bits", len(fraction_bits))
    return regime_length, exponent_bits, fraction_bits


def get_string_bits_small(abso_val, useed,es,nbits):
    
    #print("\nin get_regime_bits_small")
    
    result = abso_val
    #print("result", result)
    regime_length = 0
    while result < 1:
        result = result*useed
        #print("result*useed", result)
        regime_length += 1
    
    print("result: ", result)
    product = float(abso_val) * (useed**regime_length)
    print("product: ", product)
    exponent_bits, fraction_bits = get_exponent_and_frac_bits(product,useed,es,nbits, regime_length)
    print("nbits: ", nbits, "regime_length",regime_length, "len exponent_bits", len(exponent_bits), "len fraction_bits", len(fraction_bits))

    return regime_length, exponent_bits, fraction_bits

def create_regime_bits_string(regime_length, expo_sign_bit):
    
    #reg_str = sign_bit
    reg_str = ""
    print("expo_sign_bit", expo_sign_bit)
    bit_to_add = 1-int(expo_sign_bit)
    print("bit_to_add", bit_to_add)
    for i in range(regime_length):
        reg_str += str(bit_to_add)
        #print("reg_str", reg_str)
    reg_str += str(expo_sign_bit)
    return reg_str

#Reference: posit paper section 4 "Converting values into posits"
def convert_float_to_posit(float_value, nbits, es):
    #nbits = wordsize #es - maximum size of exponent field
    #exponent_bits, regime_bits, fraction_bits = (0,0,0)
    #TODO: Determine the above parameters dynamically from the value of the float num provided
    
    if float_value == 0:
        return create_except_case_string("0", num_bits)
    if float_value == math.inf:
        return create_except_case_string("1", num_bits)

    abso_val = float_value
    sign_bit = "0"
    if float_value < 0:
        sign_bit = "1"
        abso_val = float_value * -1

    expo_sign_bit = "0"
    if abso_val < 1:
        expo_sign_bit = "1"
    
    posit_string = ""
    useed = 2**(2**es)
    print("useed", useed)

    regime_length,exponent_bits, fraction_bits  = (0,"","")

    if(abso_val > useed):
        regime_length, exponent_bits, fraction_bits  = get_string_bits_big(abso_val, useed,es,nbits)
    elif(abso_val < 1):
        regime_length, exponent_bits, fraction_bits  = get_string_bits_small(abso_val, useed,es,nbits)
    else:
        regime_length,exponent_bits, fraction_bits = get_string_bits_regular(abso_val, useed,es,nbits)

    print("regime_length: ", regime_length)
    regime_bits_string = create_regime_bits_string(regime_length, expo_sign_bit) #TODO: check what happens when there's no regime

    print("regime_bits_string", regime_bits_string )
    print("exponent_bits", exponent_bits)
    print("fraction_bits", fraction_bits )
    final_string = sign_bit+regime_bits_string+exponent_bits+fraction_bits

    print("len final_string", len(final_string))
    return final_string


def convert_bin_to_int(bin_string):
    len_bin = len(bin_string)
    sum = 0
    for i in range(len_bin):
        power = len_bin - 1 - i
        ind_val = int(bin_string[i])
        val = ind_val * (2 ** power)
        sum += val
    return sum

def convert_bin_to_fraction(bin_string):
    len_bin = len(bin_string)
    sum = 0
    for i in range(len_bin):
        power = i+1
        ind_val = int(bin_string[i])
        val = ind_val * (float(1) /(2** power))
        sum += val
    return sum

def all_zero(posit_str):
    for i in range(len(posit_str)):
        if int(posit_str[i]) == 1: return False
    return True

def twos_complement(posit_string):
    #TODO: implement twos_complement
    return posit_string

#Reference: posit paper section 1.2
def convert_posit_to_float(es, posit_string):
    
    if all_zero(posit_string):
        print("expression_val", 0)
        return 0
    
    if int(posit_string[0]) == 1 and all_zero(posit_string[1:]):
        print("expression_val", math.inf)
        return math.inf

    #TODO: Extract regime bits dynamically
    sign = -1 if int(posit_string[0]) == 1 else 1
    
    if (sign == -1):
       posit_string = twos_complement(posit_string)
    #print("es:", es)
    scale_power_or_useed = 2**(2**es)
    #print("scale_power_or_useed:", scale_power_or_useed)
    scale_sign =  -1 if int(posit_string[es+1]) == 1 else 1
    #print("scale_sign * es: ", scale_sign * es)
    first_scale_factor = scale_power_or_useed ** (scale_sign * es)
    #print("first_scale_factor:", first_scale_factor)
    
    end_exponent_bits_index = es+2+es
    #print("end_exponent_bits_index", end_exponent_bits_index)
    
    exponent_bits = posit_string[int(es+2):int(end_exponent_bits_index)]
    #print("exponent_bits: ", exponent_bits)

    exponent_val = convert_bin_to_int(exponent_bits)
    second_scale_factor = 2**exponent_val
    fraction_bits = posit_string[end_exponent_bits_index:len(posit_string)]
    #print("fraction_bits: ", fraction_bits)

    fraction_val = convert_bin_to_fraction(fraction_bits)
    #print("fraction_val: ", fraction_val)

    fraction_factor = (1 + float(fraction_val))
    #print("fraction_factor: ", fraction_factor)
    
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
    fp_sign = ""
    abso_val = float_num
    if float_num < 0:
        fp_sign += "1"
        abso_val = abso_val * -1
    else:
        fp_sign += "0"

    fractional_part = 0
    integral_part = 0
    if abso_val < 1:
        fractional_part = abso_val
    else:
        integral_part = int(abso_val)
        fractional_part = abso_val - integral_part
    print("fractional_part: ", fractional_part, "integral_part", integral_part )

    int_bin_string = convert_int_to_bin(integral_part, config_num_int_bits)
    frac_bin_string = convert_fractional_decimal_to_bin(fractional_part, num_frac_bits)


    return fp_sign + int_bin_string +"." + frac_bin_string

def test_conversions2():
    
    test_numbers = [3.553926946732666e-06]
    num = 3.553926946732666e-06
    for i in range(15):
        posit_value = convert_float_to_posit( num, posit_num_bits, i)
        print("onbtained posit value: ", posit_value,"\n" )

def test_conversions():

    #test_numbers = [3.553926944732666e-06, 3.553926946732666e-06, 3.553926946732666e-04]
    test_numbers = [3.55393e-06, 3.553926944732666e-06, 3.553926946732666e-06, 2.08975757675, -0.755343]

    for i in range(len(test_numbers)):
        print("\n-------------------------")
        num = test_numbers[i]
        print("converting float ", num, " to posit" )
        posit_value = convert_float_to_posit( num, posit_num_bits, 3)
        print("onbtained posit value: ", posit_value )
        print("converting posit ",posit_value, "back to float " )
        float_point_value = convert_posit_to_float(3, posit_value)
        print("onbtained float point value: ", float_point_value )
        print("diff in floats before & after posit conversion: ", num - float_point_value)
        print("-------------------------\n")
        print("converting float:", num, " to fixed point num")
        fixed_string = convert_float_to_fixed_point(num)
        
        print("fixed_point_string: ", fixed_string)
        print("converting fixed point num:", fixed_string, " to float")
        decimal_float = convert_fixed_point_to_float(fixed_string)
        print("float: ", decimal_float)
        print("diff in floats before & after fixed-point conversion: ", num - decimal_float)
        print("-------------------------\n")


test_conversions()

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
