from django.shortcuts import render, redirect

# Create your views here.

from django.http import HttpResponse
from .models import Document
from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm, QueryForm
from django.template import loader
# Imaginary function to handle an uploaded file.
from .utils import generate_vectors, ask_query
from django.views.generic.edit import FormView
from .forms import FileFieldForm

global vector_data
vector_data = []


def index(request):
    docs = Document.objects.all()
    ans = []
    for doc in docs:
        ans.append(doc.title)
    return HttpResponse(f"{ans}")


def submit_query(request):
    form = QueryForm()
    return render(request, 'query.html', {'form': form})


class FileFieldFormView(FormView):
    form_class = FileFieldForm
    template_name = "upload.html"  # Replace with your template.
    success_url = "..."  # Replace with your URL or reverse().

    def form_valid(self, form):
        files = form.cleaned_data["file_field"]

        return super().form_valid(form)

    # def upload_docs(self, request):
    #     return render(request, 'upload.html', {'form': form})


def get_trained_on_documents(request):
    form = FileFieldForm(request.POST, request.FILES)
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        for f in files:
            obj = Document.objects.create(file=f)
        generate_vectors()
        print("I am getting executed")
        return redirect('submit-query')
    context = {'form': form}
    return render(request, "upload.html", context)


def get_answer(request):
    if request.method == 'POST':
        form = QueryForm(request.POST)
        result = ask_query(request.POST.get("query"))

        if form.is_valid():
            return HttpResponseRedirect(f'{form.query}, <br>{result["result"]}, <br>{result["source_documents"]}')
    else:
        form = QueryForm()

    return render(request, 'query.html', {'form': form})
