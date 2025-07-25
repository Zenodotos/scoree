from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.db.models import Q, Count, Prefetch
from django.views.generic import ListView, DetailView
from django.views import View
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db import transaction
from django.utils import timezone
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import unicodedata
import re
import pathlib

from .models import Patient, Visit, PatientDiagnosis, VisitDiagnosis, Diagnosis
from score2.models import Score2Result


class PatientListView(ListView):
    model = Patient
    template_name = 'patients/patient_list.html'
    context_object_name = 'patients'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Patient.objects.select_related().prefetch_related(
            'visits',
            'score2_results'
        ).annotate(
            visits_count=Count('visits'),
            score2_count=Count('score2_results')
        )
        
        # Search functionality
        search_query = self.request.GET.get('search')
        if search_query:
            queryset = queryset.filter(
                Q(pesel__icontains=search_query) |
                Q(full_name__icontains=search_query)
            )
        
        # Filter by diabetes status
        diabetes_filter = self.request.GET.get('diabetes')
        if diabetes_filter in ('yes', 'no'):
            diabetes_codes = ['E10', 'E11', 'E13', 'E14']

            # zbuduj Q() testujące każdy kod (zarówno chronic, jak i visits)
            diabetes_q = Q()
            for code in diabetes_codes:
                diabetes_q |= Q(chronic_diagnoses__diagnosis_code__startswith=code)
                diabetes_q |= Q(visits__diagnoses__diagnosis_code__startswith=code)

            if diabetes_filter == 'yes':
                queryset = queryset.filter(diabetes_q).distinct()
            else:  # 'no'
                queryset = queryset.exclude(diabetes_q)
        
        # Filter by age range
        age_filter = self.request.GET.get('age')
        if age_filter:
            today = date.today()
            if age_filter == '40-49':
                start_date = today - relativedelta(years=50)
                end_date = today - relativedelta(years=40)
            elif age_filter == '50-59':
                start_date = today - relativedelta(years=60)
                end_date = today - relativedelta(years=50)
            elif age_filter == '60-69':
                start_date = today - relativedelta(years=70)
                end_date = today - relativedelta(years=60)
            elif age_filter == '70+':
                start_date = date(1900, 1, 1)
                end_date = today - relativedelta(years=70)
            else:
                start_date = None
                end_date = None
            
            if start_date and end_date:
                queryset = queryset.filter(
                    date_of_birth__gte=start_date,
                    date_of_birth__lt=end_date
                )
        
        # Filter by SCORE2 calculation status
        score_filter = self.request.GET.get('score_status')
        if score_filter == 'calculated':
            queryset = queryset.filter(score2_results__isnull=False).distinct()
        elif score_filter == 'not_calculated':
            queryset = queryset.filter(score2_results__isnull=True)
        
        return queryset.order_by('-updated_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Add summary statistics
        total_patients = Patient.objects.count()
        patients_with_visits = Patient.objects.filter(visits__isnull=False).distinct().count()
        patients_with_scores = Patient.objects.filter(score2_results__isnull=False).distinct().count()
        
        # Diabetes statistics
        diabetes_codes = ['E10', 'E11', 'E13', 'E14']
        diabetic_patients = Patient.objects.filter(
            Q(chronic_diagnoses__diagnosis_code__startswith__in=diabetes_codes) |
            Q(visits__diagnoses__diagnosis_code__startswith__in=diabetes_codes)
        ).distinct().count()
        
        context.update({
            'total_patients': total_patients,
            'patients_with_visits': patients_with_visits,
            'patients_with_scores': patients_with_scores,
            'diabetic_patients': diabetic_patients,
            'search_query': self.request.GET.get('search', ''),
            'diabetes_filter': self.request.GET.get('diabetes', ''),
            'age_filter': self.request.GET.get('age', ''),
            'score_filter': self.request.GET.get('score_status', ''),
        })
        
        return context


class PatientDetailView(DetailView):
    model = Patient
    template_name = 'patients/patient_detail.html'
    context_object_name = 'patient'
    
    def get_object(self):
        return get_object_or_404(
            Patient.objects.prefetch_related(
                'visits__diagnoses',
                'chronic_diagnoses',
                Prefetch(
                    'score2_results',
                    queryset=Score2Result.objects.select_related('visit').order_by('-created_at')
                )
            ),
            pk=self.kwargs['pk']
        )
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        patient = self.object
        
        # Get visits ordered by date
        visits = patient.visits.all().order_by('-visit_date')
        
        # Get SCORE2 results grouped by visit
        score_results = {}
        for result in patient.score2_results.all():
            visit_id = result.visit.id
            if visit_id not in score_results:
                score_results[visit_id] = []
            score_results[visit_id].append(result)
        
        # Prepare visit data with score results
        visit_data = []
        for visit in visits:
            visit_scores = score_results.get(visit.id, [])
            visit_data.append({
                'visit': visit,
                'scores': visit_scores,
                'has_score': len(visit_scores) > 0
            })
        
        # Get smoking status
        smoking_status, smoking_info = patient.get_smoking_status()
        
        # Get diabetes info
        has_diabetes = patient.has_diabetes()
        diabetes_age = patient.get_diabetes_age_at_diagnosis()
        
        # Calculate what scores are possible for latest visit
        latest_visit = patient.get_latest_visit()
        possible_scores = []
        missing_data = []
        
        if latest_visit:
            age = patient.calculate_age(latest_visit.visit_date)
            
            # Check data availability
            has_sbp = latest_visit.systolic_pressure is not None
            has_total_chol = latest_visit.cholesterol_total is not None
            has_hdl_chol = latest_visit.cholesterol_hdl is not None
            has_hba1c = latest_visit.hba1c is not None
            has_egfr = latest_visit.egfr is not None
            
            if not has_sbp:
                missing_data.append('Ciśnienie skurczowe')
            if not has_total_chol:
                missing_data.append('Cholesterol całkowity')
            if not has_hdl_chol:
                missing_data.append('Cholesterol HDL')
            
            # Determine possible score types
            if 40 <= age <= 69 and not has_diabetes and has_sbp and has_total_chol and has_hdl_chol:
                possible_scores.append('SCORE2')
            elif 40 <= age <= 69 and has_diabetes and has_sbp and has_total_chol and has_hdl_chol:
                if diabetes_age and has_hba1c and has_egfr:
                    possible_scores.append('SCORE2-Diabetes')
                else:
                    if not diabetes_age:
                        missing_data.append('Wiek w momencie diagnozy cukrzycy')
                    if not has_hba1c:
                        missing_data.append('HbA1c')
                    if not has_egfr:
                        missing_data.append('eGFR')
            elif 70 <= age <= 89 and has_sbp and has_total_chol and has_hdl_chol:
                possible_scores.append('SCORE2-OP')
            elif age < 40:
                missing_data.append('Wiek poniżej 40 lat (minimum dla wszystkich skal)')
            elif age > 89:
                missing_data.append('Wiek powyżej 89 lat (maksimum dla SCORE2-OP)')
        
        context.update({
            'visit_data': visit_data,
            'smoking_status': smoking_status,
            'smoking_info': smoking_info,
            'has_diabetes': has_diabetes,
            'diabetes_age': diabetes_age,
            'latest_visit': latest_visit,
            'possible_scores': possible_scores,
            'missing_data': missing_data,
        })
        
        return context


class ImportDataView(View):
    template_name = 'patients/import_data.html'
    
    def get(self, request):
        return render(request, self.template_name)
    
    def post(self, request):
        if 'file' not in request.FILES:
            messages.error(request, 'Nie wybrano pliku do importu.')
            return render(request, self.template_name)
        
        uploaded_file = request.FILES['file']
        
        # Check file extension
        if not uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            messages.error(request, 'Obsługiwane są tylko pliki Excel (.xlsx, .xls).')
            return render(request, self.template_name)
        
        try:
            # Process the uploaded file
            results = self._process_excel_file(uploaded_file)
            
            messages.success(
                request, 
                f'Import zakończony! Przetworzono {results["patients_processed"]} pacjentów, '
                f'{results["visits_processed"]} wizyt, {results["diagnoses_processed"]} diagnoz.'
            )
            
            return redirect('patients:patient_list')
            
        except Exception as e:
            messages.error(request, f'Błąd podczas importu: {str(e)}')
            return render(request, self.template_name)
    
    def _process_excel_file(self, uploaded_file):
        """Process Excel file similar to the seeder script"""
        
        # Column mapping (same as in seeder)
        COLS = {
            "PACJENT": "full_name",
            "IDENTYFIAKTOR": "pesel",
            "DATA URODZENIA": "dob",
            "ROZPOZNANIE Z WIZYTY": "visit_dx",
            "DATA OSTATNIEJ WIZYTY": "visit_date",
            "ROZPOZNANIE PRZEWLEKŁE": "chronic_dx",
            "ADRES": "address",
            "TEL. KOMÓRKOWY": "phone_mobile",
            "TEL. STACJONARNY": "phone_landline",
            "DATA ROZPOZNANIA SCHORZENIA PRZEWLEKłEGO": "chronic_dx_date",
            "DATA OSTATNIEJ WIZYTY ZE SCHORZENIEM PRZEWLEKŁYM": "last_chronic_visit",
            "ŚR. CIśNIENIE SKURCZOWE": "systolic_pressure",
            "HEMOGLOBINA GLIKOWANA": "hba1c",
            "EGFR": "egfr",
            "CHOLESTEROL CAŁKOWITY": "cholesterol_total",
            "CHOLESTEROL HDL": "cholesterol_hdl",
        }
        
        # Read Excel file
        df_raw = pd.read_excel(uploaded_file)
        
        # Build rename map
        rename_map = self._build_rename_map(df_raw.columns.tolist(), COLS)
        
        # Rename columns
        df = df_raw.rename(columns=rename_map)
        
        # Check for missing columns
        missing = [v for v in COLS.values() if v not in df.columns]
        if missing:
            raise ValueError(f"Brakuje kolumn: {missing}")
        
        # Select only needed columns
        df = df.loc[:, list(COLS.values())]
        
        # Clean PESEL column
        df["pesel"] = df["pesel"].apply(lambda x: "" if pd.isna(x) else str(x))
        
        # Convert date columns
        for c in ("dob", "visit_date", "chronic_dx_date", "last_chronic_visit"):
            df[c] = df[c].apply(self._safe_date)
        
        # Handle NULL values for other columns
        for c in (
            "address", "phone_mobile", "phone_landline",
            "visit_dx", "chronic_dx",
            "systolic_pressure", "hba1c", "egfr",
            "cholesterol_total", "cholesterol_hdl"
        ):
            df[c] = df[c].apply(lambda x: None if pd.isna(x) else x)
        
        # Filter out rows with empty PESEL
        df = df[df["pesel"] != ""]
        
        # Process patients
        results = {
            'patients_processed': 0,
            'visits_processed': 0,
            'diagnoses_processed': 0
        }
        
        with transaction.atomic():
            for pesel, patient_group in df.groupby('pesel'):
                try:
                    patient_results = self._process_patient_group(patient_group)
                    results['patients_processed'] += 1
                    results['visits_processed'] += patient_results.get('visits', 0)
                    results['diagnoses_processed'] += patient_results.get('diagnoses', 0)
                except Exception as e:
                    print(f"Error processing patient {pesel}: {str(e)}")
                    continue
        
        return results
    
    def _canonical(self, s: str) -> str:
        """Normalize string for column matching"""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.upper()
        return re.sub(r"[^A-Z0-9]", "", s)
    
    def _build_rename_map(self, raw_cols, template):
        """Build column rename mapping"""
        canon_to_target = {self._canonical(k): v for k, v in template.items()}
        rename_map = {}
        for raw in raw_cols:
            c = self._canonical(raw)
            if c in canon_to_target:
                rename_map[raw] = canon_to_target[c]
        return rename_map
    
    def _pesel_to_gender(self, pv) -> str:
        """Extract gender from PESEL"""
        return "M" if int(str(pv)[-2]) % 2 else "F"
    
    def _safe_date(self, val):
        """Safely convert to date"""
        if pd.isna(val):
            return None
        try:
            return pd.to_datetime(val, dayfirst=True, errors="coerce").date()
        except:
            return None
    
    def _process_patient_group(self, patient_group):
        """Process all rows for a single patient"""
        first_row = patient_group.iloc[0]
        
        # Create or update patient
        patient, created = Patient.objects.get_or_create(
            pesel=first_row.pesel,
            defaults={
                'full_name': first_row.full_name,
                'date_of_birth': first_row.dob,
                'gender': self._pesel_to_gender(first_row.pesel),
                'address': first_row.address,
                'phone_mobile': first_row.phone_mobile,
                'phone_landline': first_row.phone_landline,
            }
        )
        
        if not created:
            # Update existing patient data (except basic info)
            if first_row.address:
                patient.address = first_row.address
            if first_row.phone_mobile:
                patient.phone_mobile = first_row.phone_mobile
            if first_row.phone_landline:
                patient.phone_landline = first_row.phone_landline
            patient.save()
        
        results = {'visits': 0, 'diagnoses': 0}
        
        # Create visit if visit_date exists and visit doesn't exist yet
        if first_row.visit_date:
            visit, visit_created = Visit.objects.get_or_create(
                patient=patient,
                visit_date=first_row.visit_date,
                defaults={
                    'systolic_pressure': first_row.systolic_pressure,
                    'hba1c': first_row.hba1c,
                    'egfr': first_row.egfr,
                    'cholesterol_total': first_row.cholesterol_total,
                    'cholesterol_hdl': first_row.cholesterol_hdl,
                }
            )
            
            if visit_created:
                results['visits'] = 1
            
            # Add visit diagnoses
            if first_row.visit_dx:
                for code in str(first_row.visit_dx).split(','):
                    code = code.strip()
                    if code:
                        self._ensure_diagnosis(code)
                        VisitDiagnosis.objects.get_or_create(
                            visit=visit,
                            diagnosis_code=code
                        )
        
        # Process chronic diagnoses
        for _, row in patient_group.iterrows():
            if row.chronic_dx:
                self._ensure_diagnosis(row.chronic_dx)
                
                age_at = None
                if row.chronic_dx_date and row.dob:
                    age_at = relativedelta(row.chronic_dx_date, row.dob).years
                
                diagnosis, diag_created = PatientDiagnosis.objects.get_or_create(
                    patient=patient,
                    diagnosis_code=row.chronic_dx,
                    defaults={
                        'diagnosed_at': row.chronic_dx_date,
                        'last_visit_with_condition': row.last_chronic_visit,
                        'age_at_diagnosis': age_at,
                    }
                )
                
                if not diag_created:
                    # Update existing diagnosis
                    if row.chronic_dx_date:
                        diagnosis.diagnosed_at = row.chronic_dx_date
                    if row.last_chronic_visit:
                        diagnosis.last_visit_with_condition = row.last_chronic_visit
                    if age_at:
                        diagnosis.age_at_diagnosis = age_at
                    diagnosis.save()
                
                if diag_created:
                    results['diagnoses'] += 1
        
        return results
    
    def _ensure_diagnosis(self, code):
        """Ensure diagnosis code exists in database"""
        if code:
            Diagnosis.objects.get_or_create(code=code)


class PatientSearchView(View):
    """AJAX view for patient search autocomplete"""
    
    def get(self, request):
        query = request.GET.get('q', '')
        if len(query) < 2:
            return JsonResponse({'results': []})
        
        patients = Patient.objects.filter(
            Q(pesel__icontains=query) |
            Q(full_name__icontains=query)
        )[:10]
        
        results = []
        for patient in patients:
            results.append({
                'id': patient.id,
                'text': f"{patient.full_name or patient.pesel} ({patient.pesel})",
                'age': patient.age,
                'has_diabetes': patient.has_diabetes(),
            })
        
        return JsonResponse({'results': results})