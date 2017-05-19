from celery import Celery
import os
from classify import classify
from confusion_matrix import create_confusion_matrix

app = Celery('tasks', broker='pyamqp://admin:admin@localhost//svm', backend='redis://localhost:6379/0')


@app.task
def start_classify(trainDir, testDir, pathName, train, counter, alg):
    try:
        if not train:
            #classify(testDir, path='%s_%s' % (pathName, 'test_score'), counter=counter, alg=alg)
            pass
        if train:
            create_confusion_matrix(trainDir, testDir,
                                    out_dir=os.path.abspath(os.path.join(trainDir)), path_name=pathName,
                                    counter=counter, alg=alg)
            #classify(trainDir, path='%s_%s' % (pathName, 'train_score'), counter=counter, alg=alg)
    except Exception as e:
        print e.message