import unittest
from kostya_rules import extract_description
from kostya_regression_cases import REGRESSION_CASES
from vahta_extensions import extract,format_rate_amount
class Rules(unittest.TestCase):
 def test_kostya_original(self):
  for f,text,value,status in REGRESSION_CASES:
   with self.subTest(feature=f,text=text):
    r=extract_description(text)[f];self.assertEqual((r['value'],r['status']),(value,status))
 def test_presence_not_paid_hours(self):self.assertEqual(extract('Смена с 08:00 до 20:00.')['paid_hours']['status'],'not_stated')
 def test_paid_hours(self):self.assertEqual(extract('К оплате 11 часов за смену.')['paid_hours']['value'],11)
 def test_weekly_not_intervahta(self):self.assertEqual(extract('График 6/1.')['rest_duration']['status'],'not_stated')
 def test_slash_not_rest(self):self.assertEqual(extract('Вахта 15/30 дней.')['rest_duration']['status'],'not_stated')
 def test_rest(self):self.assertEqual(extract('После вахты 15 дней отдыха.')['rest_duration']['value'],'15 days')
 def test_work_shifts_not_days(self):self.assertEqual(extract('Вахта 30 смен.')['vahta_work_duration']['value'],'30 work_shifts')
 def test_address_not_publication(self):self.assertEqual(extract('Адрес работы: Московская область, Софьино.')['workplace']['status'],'resolved')
 def test_rate_tax(self):
  r=extract('Оплата 4 000 рублей за смену на руки.')['rate_options']['value'];self.assertEqual((r['low'],r['unit'],r['tax']),(4000,'shift','net'))
 def test_piecework_not_unconditional(self):self.assertTrue(extract('Оплата 5000 рублей за смену при выполнении нормы.')['rate_options']['value']['conditional'])
 def test_monthly_body_rate(self):
  r=extract('Оплата: 140 000 руб./мес. на руки.')['rate_options']['value'];self.assertEqual(r['unit'],'month')
 def test_duration_qualified_shift(self):
  r=extract('3 500 рублей за 11-часовую смену.')['rate_options']['value'];self.assertEqual((r['low'],r['unit']),(3500,'shift'))
 def test_multiday_salary_not_shift(self):
  r=extract('З/плата от 130.000 до 140.000 на руки за 15 дней.')['non_shift_amounts'];self.assertEqual(r['value'],140000)
 def test_rotation_salary(self):
  r=extract('ЗП за вахту 3 месяца 515000-00.')['non_shift_amounts'];self.assertEqual(r['value'],515000)
 def test_optional_overtime_not_total_paid_hours(self):
  r=extract('Продолжительность смены 12 часов.\nПодработки по желанию:\nСмена 18 часов оплачивается -4800.')
  self.assertEqual(r['paid_hours']['status'],'not_stated');self.assertEqual(r['paid_hours_rejected']['value'],18)
 def test_paid_break_not_total_paid_hours(self):
  self.assertEqual(extract('Смены по 11 часов (1 час оплачиваемого обеда).')['paid_hours']['status'],'not_stated')
 def test_overtime_threshold_not_total_paid_hours(self):
  self.assertEqual(extract('Переработки свыше 11 часов оплачиваются дополнительно.')['paid_hours']['status'],'not_stated')
 def test_explicit_paid_hours_inside_shift_preserved(self):
  self.assertEqual(extract('Из 12 часов смены 11 часов оплачиваются. Предусмотрен обед.')['paid_hours']['value'],11)
 def test_supplement_not_base_salary(self):
  r=extract('Доплата за заправку - 170 ₽ на руки за полную заправку (в среднем 7 000 ₽ в месяц).')['rate_options']['value']
  self.assertTrue(r['bonus_context'])
 def test_base_rate_after_support_not_supplement(self):
  r=extract('Бесплатное жилье, компенсация проезда, ставка 5300 ₽ за смену.')['rate_options']['value']
  self.assertFalse(r['bonus_context'])
 def test_slash_duration_options_not_last_number(self):
  for text in ['Вахта 20/33/45/90 смен на выбор.','Вахта 15/30/45/60/90 смен.','Вахта 90/30 дней.','Вахта 30/30 дней.','Вахта 30–45 дней.','Вахта 20,35,45,60,70 смен.','Вахта 90,120 дней.','Вахта 15 или 30 смен.','Вахта от 30 до 45 дней.']:
   with self.subTest(text=text):self.assertEqual(extract(text)['vahta_work_duration']['status'],'ambiguous')
 def test_decimal_month_and_explicit_rest_not_mixed(self):
  r=extract('Вахтовый метод работы: 2,5 месяца/1 месяц отдыха;')
  self.assertEqual(r['vahta_work_duration']['value'],'2.5 months');self.assertEqual(r['rest_duration']['value'],'1 months')
 def test_recovered_rate_keeps_lower_bound(self):
  for shape,low,high,expected in [('from',200000,200000,'от 200000'),('exact',200000,200000,'200000'),('range',3000,4000,'3000 — 4000')]:
   with self.subTest(shape=shape):self.assertEqual(format_rate_amount({'shape':shape,'low':low,'high':high}),expected)
if __name__=='__main__':unittest.main()
