import cv2

image = cv2.imread('WAC_TIO2_GLOBAL_MAP.png',0)

mask = cv2.imread('LROC_GLOBAL_MARE_180_WAC_MASK.png',0)
mask_hand = cv2.imread('LROC_GLOBAL_MARE_180_WAC_MASK_HAND.png',0)
mask_neg = cv2.imread('LROC_GLOBAL_MARE_180_WAC_MASK_NEG.png',0)

result = cv2.bitwise_and(image, mask)
result[mask==0] = 22 # set background to 1 w%

result_omit = cv2.bitwise_and(image, mask_neg)
result_omit[result<22] = 255 # mark what will be set to 1 w% from below 1 w%

result[result<22] = 22 # set everything below 1% to 1 w%


result_hand = cv2.bitwise_and(image, mask_hand)
result_hand[mask_hand==0] = 22 # set background to 1 w%
result_hand[result_hand<22] = 22 # set everything below 1% to 1 w%


cv2.imwrite('WAC_TIO2_GLOBAL_MASKED_MAP.png', result)
cv2.imwrite('WAC_TIO2_GLOBAL_MASKED_MAP_HAND.png', result_hand)
cv2.imwrite('WAC_TIO2_GLOBAL_MASKED_OMITTED.png', result_omit)

print('COMPLETED.')
