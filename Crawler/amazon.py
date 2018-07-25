import urllib.request
import re
import csv

headers = {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6',"strict":False}
#url = "https://www.amazon.com/SkyTech-ArchAngel-Gaming-Computer-Desktop/product-reviews/B01M3UKNVD/ref=cm_cr_getr_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber="

pattern = r'<i data-hook="review-star-rating".*?><span class="a-icon-alt">(.*?)out of.*?</span>.*?<a data-hook="review-title".*?>(.*?)</a>.*?<span data-hook="review-body".*?>(.*?)</span>'
pattern_re = re.compile(pattern,re.S)

pattern_page = r'>(\d+)</a></li><li class="a-last">'
pattern_page_re = re.compile(pattern_page,re.S)

def find(url):
    pnum = 1
    result = []

    url2 = url+'1'
    request = urllib.request.Request(url2,headers=headers)
    response = urllib.request.urlopen(request)
    page = response.read().decode("latin_1")
    #max = 4
    max1 = pattern_page_re.findall(page)
    max=max1[0]

    while True:
        url1 = url+str(pnum)

        request = urllib.request.Request(url1, headers=headers)
        sub_page_orig = urllib.request.urlopen(request)
        sub_page = sub_page_orig.read().decode("latin_1")
        items = pattern_re.findall(sub_page)

        for item in items :
            result.append([item[0],item[1],item[2]])

        print("你小子这么慢才到%d" % pnum)
        pnum += 1
        if pnum > int(max):
            break
    return result

res=["https://www.amazon.com/SkyTech-ArchAngel-Gaming-Computer-Desktop/product-reviews/B01M3UKNVD/ref=cm_cr_getr_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Dell-Computer-Professional-Certified-Refurbished/product-reviews/B00NIYVRXE/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Dell-OptiPlex-Desktop-Complete-Computer/product-reviews/B00IOTZGOE/ref=cm_cr_arp_d_viewpnt_lft?ie=UTF8&reviewerType=avp_only_reviews&filterByStar=positive&pageNumber=",
     "https://www.amazon.com/Performance-RCA-Touchscreen-Quad-Core-Processor/product-reviews/B01MQD63WX/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Chromebook-C202SA-YS02-Ruggedized-Resistant-Celeron/product-reviews/B01DBGVB7K/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Dell-OptiPlex-Professional-Certified-Refurbished/product-reviews/B00J0ETHCY/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/HP-Pavilion-22cwa-21-5-inch-Backlit/product-reviews/B015WCV70W/ref=cm_cr_arp_d_viewpnt_lft?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Acer-Chromebook-CB3-131-C3SZ-11-6-Inch-Dual-Core/product-reviews/B019G7VPTC/ref=cm_cr_arp_d_viewpnt_lft?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Acer-Chromebook-Convertible-11-6-Inch-CB5-132T-C1LK/product-reviews/B01J42JPJG/ref=cm_cr_arp_d_viewpnt_lft?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Apple-Macbook-MB403LL-Laptop-Mobile/product-reviews/B00159LRXY/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/SkyTech-ArchAngel-Gaming-Computer-Desktop/product-reviews/B01M3UKNVD/ref=cm_cr_getr_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/HP-14-inch-E2-7110-Windows-14-an013nr/product-reviews/B01F4ZG68A/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Lightweight-11-6-inch-Quad-Core-Microsoft-Subscription/product-reviews/B01LT692RK/ref=cm_cr_arp_d_viewpnt_lft?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Acer-E5-575-33BM-15-6-Inch-Processor-Generation/product-reviews/B01K1IO3QW/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/HP-15-F222WM-Pentium-Processor-Windows/product-reviews/B01MRS3MIS/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Lenovo-Performance-Dual-Core-Processor-Bluetooth/product-reviews/B01MQTJXWZ/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/HP-Stream-14-ax010nr-Celeron-Personal/product-reviews/B01JLCKP34/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Dell-15-6-Inch-Touchscreen-RealSense-Bluetooth/product-reviews/B01D27ERMO/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/C201PA-DS02-Chromebook-1-8GHz-Quad-Core-LPDDR3/product-reviews/B00VUV0MG0/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Acer-E5-575G-57D4-15-6-Inches-Notebook-i5-7200U/product-reviews/B01LD4MGY4/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/VivoBook-15-6-Inch-Performance-Premium-Processor/product-reviews/B01MQEI5SW/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/HP-Pavilion-15-15-6-Inch-Quad-Core/product-reviews/B01EMWKFEC/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/HP-Stream-Premium-Touchscreen-Laptop/product-reviews/B015CQ8PNA/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/HP-Notebook-15-ay011nr-15-6-Inch-Processor/product-reviews/B01CGGOZOM/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Acer-Aspire-E5-575G-53VG-15-6-Inch-Windows/product-reviews/B01DT4A2R4/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Dell-Inspiron-Touchscreen-Signature-Bluetooth/product-reviews/B01N0K3246/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/HP-15-af131dx-P1A95UA-Quad-Core-A6-5200/product-reviews/B017QDHENY/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Acer-Chromebook-15-6-inch-Celeron-CB5-571-C4G4/product-reviews/B01CMYGAGY/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Dell-15-6-Inch-Quad-Core-i5-6300HQ-Processor/product-reviews/B015PYYDMQ/ref=cm_cr_getr_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/C100PA-DB02-10-1-inch-Chromebook-1-8GHz-Operation/product-reviews/B00ZS4HK0Q/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/HP-Flagship-15-ay191ms-Touchscreen-Signature/product-reviews/B00SL6A8NY/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Acer-Aspire-i3-6100U-Windows-ES1-572-31KW/product-reviews/B01J42JPL4/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Samsung-Chromebook-XE500C13-K01US-16GB-Laptop/product-reviews/B01APA6K6M/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Amazons-Choice-UX330UA-AH54-Ultra-Slim-Fingerprint/product-reviews/B01M18UZF5/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Acer-Chromebook-Aluminum-Quad-Core-CB3-431-C5FM/product-reviews/B01CVOLVPA/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Samsung-Chromebook-Wi-Fi-11-6-Inch-Refurbished/product-reviews/B00M9K7L8S/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Chromebook-C302CA-DHM4-12-5-Inch-Touchscreen-storage/product-reviews/B01N5G5PG2/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Dell-i7559-7512GRY-Touchscreen-Generation-Processor/product-reviews/B015PZ0EHS/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber=",
     "https://www.amazon.com/Apple-Macbook-MB403LL-Laptop-Mobile/product-reviews/B00159LRXY/ref=cm_cr_arp_d_show_all?ie=UTF8&reviewerType=all_reviews&pageNumber="]


i = 1
for url in res:
    print('这小伙子开始偷第%d个产品' %i)
    #print(url)
    result=find(url)
    i+=1
    with open('/Users/hhy/Desktop/pa1.csv', 'a', encoding='gb18030', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)