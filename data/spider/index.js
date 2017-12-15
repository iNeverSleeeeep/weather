var http = require('http');
var fs = require('fs');
var cheerio = require('cheerio');
var request = require('request');
var iconv = require('iconv-lite');
var csv = require('csv');
var async = require('async');
var _ = require('underscore')._;

var years = [2011, 2012, 2013, 2014, 2015, 2016, 2017];
var months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"];
var alldata = [];
function startRequest(url, cb) {
    console.log(url);
    http.get(url, function (res) {
        var chunks = [];

        res.on('data', function (chunk) {
            chunks.push(chunk);
        });

        res.on('end', function () {
            var html = iconv.decode(Buffer.concat(chunks), 'gb2312');
            var $ = cheerio.load(html); //采用cheerio模块解析html
            $('table').find('tr').each(function (i, tr) {
                if (i === 0) return;
                var data = [];
                $(tr).find('td').each(function (j, td) {
                    var text = $(td).text();
                    text = text.replace(/\\n/g, "").replace(/\\r/g, "").replace(/\s/g, "");
                    if (j === 0)
                    {
                        data.push(parseInt(text.substr(0, 4)+text.substr(5, 2)+text.substr(8, 2)));
                    }
                    else if (j === 1)
                    {
                        data.push(text.substring(0, text.indexOf('/')));
                    }
                    else if (j === 2)
                    {
                        var index = text.indexOf('℃');
                        var last = text.lastIndexOf('℃');
                        data.push(text.substring(0, index));
                        data.push(text.substring(index+2, last));
                    }
                    else if (j === 3)
                    {
                        data.push(text.substring(0, text.indexOf('风')+1));
                        index = text.indexOf('级');
                        if (index === -1)
                            data.push(0);
                        else
                            data.push(text.substr(index-1, 1));
                    }
                });
                alldata.push(data);
            });

            cb();
        });
    });
}

var urls = [];
_.each(years, function (year) {
    _.each(months, function (month) {
        urls.push("http://www.tianqihoubao.com/lishi/chengdu/month/"+year+month+".html");
    });
});

async.mapLimit(urls, 12, function (url, cb) {
    startRequest(url, cb);
}, function (err) {
    var sorted = _.sortBy(alldata, function(data){return data[0];});

    var all = [];
    _.each(sorted, function(data){all.push(data.join(','))});

    fs.writeFile('../chengdu.csv', all.join('\n'), function () {
        process.exit();
    });
});